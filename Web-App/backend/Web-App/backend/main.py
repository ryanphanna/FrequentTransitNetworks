from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import zipfile
import pandas as pd
import tempfile
import os
from collections import defaultdict
from typing import List, Tuple, Dict, Any
import hashlib
import datetime
import json
from celery import Celery

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Celery setup
broker_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
celery_app = Celery('gtfs_tasks', broker=broker_url, backend=broker_url)

# In-memory cache
cache: Dict[str, Dict[str, Any]] = {}
CACHE_MAX_AGE_DAYS = 30
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB
REQUIRED_GTFS_FILES = {'routes.txt', 'trips.txt', 'stop_times.txt', 'calendar.txt', 'shapes.txt'}

def load_feed_metadata(base_path: str) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    path = os.path.join(base_path, 'feed_info.txt')
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            row = df.iloc[0]
            metadata = {k: row.get(k, '')
                        for k in ['feed_publisher_name', 'feed_publisher_url', 'feed_version']}
            for d in ['feed_start_date', 'feed_end_date']:
                v = row.get(d)
                if pd.notna(v):
                    metadata[d] = datetime.datetime.strptime(str(int(v)), '%Y%m%d').date().isoformat()
        except Exception:
            pass
    return metadata

def validate_gtfs_zip(zip_path: str) -> None:
    size = os.path.getsize(zip_path)
    if size > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail='Uploaded file too large')
    with zipfile.ZipFile(zip_path, 'r') as z:
        names = set(z.namelist())
    missing = REQUIRED_GTFS_FILES - names
    if missing:
        raise HTTPException(status_code=422,
            detail=f'Missing required GTFS files: {", ".join(sorted(missing))}')

def parse_gtfs_zip(zip_path: str) -> Tuple[Dict[str,int], list, Dict[str,Any]]:
    validate_gtfs_zip(zip_path)
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(temp_dir)

    metadata = load_feed_metadata(temp_dir)
    end_date = metadata.get('feed_end_date')
    if end_date and datetime.date.fromisoformat(end_date) < datetime.date.today():
        raise Exception('GTFS feed expired')

    try:
        routes = pd.read_csv(os.path.join(temp_dir, 'routes.txt'))
        trips = pd.read_csv(os.path.join(temp_dir, 'trips.txt'))
        stop_times = pd.read_csv(os.path.join(temp_dir, 'stop_times.txt'))
        calendar = pd.read_csv(os.path.join(temp_dir, 'calendar.txt'))
        shapes = pd.read_csv(os.path.join(temp_dir, 'shapes.txt'))
    except Exception:
        raise Exception('Error reading GTFS files')

    svc = calendar[(calendar.monday==1)&(calendar.tuesday==1)&(calendar.wednesday==1)
                   &(calendar.thursday==1)&(calendar.friday==1)]['service_id']
    trips = trips[trips.service_id.isin(svc)]
    stop_times = stop_times[stop_times.trip_id.isin(trips.trip_id)]

    merged = trips.merge(stop_times, on='trip_id')
    merged['arrival_time'] = pd.to_timedelta(merged['arrival_time'].fillna('00:00:00'))
    merged = merged[(merged.arrival_time >= pd.to_timedelta('07:00:00')) &
                    (merged.arrival_time <= pd.to_timedelta('22:00:00'))]

    headways: Dict[str, float] = {}
    for rid, grp in merged.groupby('route_id'):
        mins = grp.arrival_time.sort_values().dt.total_seconds() / 60
        if len(mins) < 2: continue
        headways[rid] = mins.diff().dropna().max()

    buckets: Dict[str, str] = {}
    for rid, h in headways.items():
        if h <= 10: buckets[rid] = '10'
        elif h <= 15: buckets[rid] = '15'
        elif h <= 20: buckets[rid] = '20'
        elif h <= 30: buckets[rid] = '30'
        elif h <= 60: buckets[rid] = '60'
        else: buckets[rid] = 'worse'

    freq_counts = defaultdict(int)
    route_shapes = []
    for _, r in routes.iterrows():
        rid = r.route_id
        b = buckets.get(rid, 'unknown')
        freq_counts[b] += 1
        sids = trips[trips.route_id==rid].shape_id.dropna().unique()
        if sids.size:
            pts = shapes[shapes.shape_id==sids[0]].sort_values('shape_pt_sequence')
            coords = pts[['shape_pt_lat','shape_pt_lon']].values.tolist()
            route_shapes.append({'route_id': rid, 'frequency_bucket': b, 'shape': coords})

    # cleanup
    for f in os.listdir(temp_dir): os.remove(os.path.join(temp_dir, f))
    os.rmdir(temp_dir)

    return freq_counts, route_shapes, metadata

@celery_app.task(bind=True)
def compare_gtfs_task(self, files_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results = []
    for info in files_info:
        filename = info['filename']
        content = info['content']
        sha = hashlib.sha256(content).hexdigest()
        entry = cache.get(sha)
        if entry and (datetime.datetime.now() - entry['timestamp']).days <= CACHE_MAX_AGE_DAYS:
            results.append(entry['data'])
            continue

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(content)
            path = tmp.name
        try:
            freqs, shapes, meta = parse_gtfs_zip(path)
            data = {'city_name': filename.replace('.zip',''),
                    'feed_info': meta,
                    'frequencies': freqs,
                    'routes': shapes}
            cache[sha] = {'timestamp': datetime.datetime.now(), 'data': data}
            results.append(data)
        finally:
            os.remove(path)
    return results

@app.post('/compare_gtfs')
async def compare_gtfs(files: List[UploadFile] = File(...)):
    if len(files) != 2:
        raise HTTPException(status_code=400, detail='Upload exactly 2 GTFS files')

    files_info = []
    for f in files:
        content = await f.read()
        if len(content) == 0:
            raise HTTPException(status_code=422, detail=f'{f.filename} is empty')
        files_info.append({'filename': f.filename, 'content': content})

    job = compare_gtfs_task.delay(files_info)
    return JSONResponse({'task_id': job.id}, status_code=202)

@app.get('/jobs/{task_id}')
async def get_job_status(task_id: str):
    res = celery_app.AsyncResult(task_id)
    if res.state == 'PENDING':
        return JSONResponse({'status': 'PENDING'})
    if res.state == 'FAILURE':
        return JSONResponse({'status': 'FAILURE', 'error': str(res.info)}, status_code=500)
    return JSONResponse({'status': 'SUCCESS', 'result': res.result})
