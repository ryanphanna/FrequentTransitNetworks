from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import zipfile, pandas as pd, tempfile, os
from collections import defaultdict

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

REQUIRED_FILES = {'routes.txt','trips.txt','stop_times.txt','calendar.txt','shapes.txt'}

def process_gtfs(path: str):
    # Unzip
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(path, 'r') as z:
        z.extractall(temp_dir)

    # Validate
    missing = REQUIRED_FILES - set(os.listdir(temp_dir))
    if missing:
        raise HTTPException(422, f"Missing: {', '.join(missing)}")

    # Load tables
    routes     = pd.read_csv(f"{temp_dir}/routes.txt")
    trips      = pd.read_csv(f"{temp_dir}/trips.txt")
    stop_times = pd.read_csv(f"{temp_dir}/stop_times.txt")
    calendar   = pd.read_csv(f"{temp_dir}/calendar.txt")
    shapes     = pd.read_csv(f"{temp_dir}/shapes.txt")

    # Weekday filter
    weekdays = calendar[
        (calendar.monday==1)&(calendar.tuesday==1)&(calendar.wednesday==1)&
        (calendar.thursday==1)&(calendar.friday==1)
    ]['service_id']
    trips = trips[trips.service_id.isin(weekdays)]
    stop_times = stop_times[stop_times.trip_id.isin(trips.trip_id)]

    # Time window 07:00–22:00
    merged = trips.merge(stop_times, on='trip_id')
    merged['arrival_time'] = pd.to_timedelta(merged['arrival_time'].fillna('00:00:00'))
    merged = merged[(merged.arrival_time >= '07:00:00') & (merged.arrival_time <= '22:00:00')]

    # Compute headways
    headways = {}
    for rid, grp in merged.groupby('route_id'):
        mins = grp.arrival_time.sort_values().dt.total_seconds()/60
        if len(mins) > 1:
            headways[rid] = mins.diff().max()

    # Bucketize
    freq_buckets = {}
    for rid, h in headways.items():
        if   h <= 10: freq_buckets[rid] = '10'
        elif h <= 15: freq_buckets[rid] = '15'
        elif h <= 20: freq_buckets[rid] = '20'
        elif h <= 30: freq_buckets[rid] = '30'
        elif h <= 60: freq_buckets[rid] = '60'
        else:          freq_buckets[rid] = 'worse'

    # Count & shapes
    freq_counts  = defaultdict(int)
    route_shapes = []
    for _, r in routes.iterrows():
        b = freq_buckets.get(r.route_id, 'unknown')
        freq_counts[b] += 1
        sids = trips[trips.route_id==r.route_id].shape_id.dropna().unique()
        if len(sids):
            pts    = shapes[shapes.shape_id==sids[0]].sort_values('shape_pt_sequence')
            coords = pts[['shape_pt_lat','shape_pt_lon']].values.tolist()
            route_shapes.append({
                'route_id': r.route_id,
                'frequency_bucket': b,
                'shape': coords
            })

    # Cleanup
    for f in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, f))
    os.rmdir(temp_dir)

    return {'frequencies': freq_counts, 'routes': route_shapes}


@app.post("/upload_gtfs")
async def upload_gtfs(file: UploadFile = File(...)):
    if not file.filename.endswith('.zip'):
        raise HTTPException(415, "Must upload a .zip GTFS file")
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.write(await file.read())
    tmp.close()
    try:
        result = process_gtfs(tmp.name)
    except HTTPException:
        os.remove(tmp.name)
        raise
    except Exception as e:
        os.remove(tmp.name)
        raise HTTPException(500, str(e))
    os.remove(tmp.name)
    return JSONResponse(result)
