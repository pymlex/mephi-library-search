from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
from models import find_best_books, df
from fastapi.responses import JSONResponse
import time
import numpy as np


class SearchRequest(BaseModel):
    query: str
    top_k: int = 3


app = FastAPI()


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/search')
def search(req: SearchRequest):
    t0 = time.time()
    idxs = find_best_books(req.query, top_k=req.top_k)
    rows = [df.iloc[i].to_dict() for i in idxs]

    def normalize_value(v):
        if isinstance(v, (np.floating, float)):
            if np.isnan(v) or np.isinf(v):
                return None
            return float(v)
        if isinstance(v, (np.integer, int, np.int_)):
            return int(v)
        if isinstance(v, np.bool_):
            return bool(v)
        return v

    rows_clean = []
    for row in rows:
        rc = {}
        for k, v in row.items():
            rc[k] = normalize_value(v)
        rows_clean.append(rc)

    rows_enc = jsonable_encoder(rows_clean)
    t1 = time.time()
    headers = {'X-Search-Time': str(t1 - t0)}
    return JSONResponse(content=rows_enc, headers=headers)
