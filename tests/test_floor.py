import copy
import json

import numpy as np
import pytest

from perception.floor import fit, FloorMap, project
from perception.calibrate_floor import marker_centre, make_markers, record_parking


@pytest.fixture
def observations():
    matrix = np.array([[0,.001,.1],[-.001,0,.5],[0,0,1]])
    train = [[0,0],[1000,0],[1000,600],[0,600]]
    checks = [[100,100],[800,200],[400,500]]
    rows = [dict(id=i, pixel=p, floor_m=project(matrix,[p])[0].tolist())
            for i,p in enumerate(train+checks)]
    return dict(colour_size=[1280,720],flip=False,mount_note='synthetic fixture, not hardware',
                fit=rows[:4],check=rows[4:])


def test_fit_and_independent_validation(observations):
    data=fit(observations)
    assert data['validation']['max_error_m'] < 1e-6
    floor=FloorMap(data,colour_size=[1280,720],flip=False)
    np.testing.assert_allclose(floor.map([500,300]),[.4,0],atol=1e-6)
    assert floor.map([1200,300]) is None


def test_reject_bad_check_even_when_fit_is_exact(observations):
    observations['check'][0]['floor_m'][0]+=.05
    with pytest.raises(ValueError,match='rejected'): fit(observations)


def test_reject_reused_check_and_collinear_fit(observations):
    bad=copy.deepcopy(observations);bad['check'][0]['id']=0
    with pytest.raises(ValueError,match='distinct'): fit(bad)
    for i,row in enumerate(observations['fit']): row['pixel']=[i,i]
    with pytest.raises(ValueError,match='line'): fit(observations)


def test_reject_tampered_matrix_and_changed_camera(observations):
    data=fit(observations)
    with pytest.raises(ValueError,match='size/flip'): FloorMap(data,colour_size=[640,480],flip=False)
    data['matrix'][0][2]+=.1
    with pytest.raises(ValueError,match='evidence'): FloorMap(data,colour_size=[1280,720],flip=False)


def test_marker_centre_is_projective_not_corner_mean():
    corners=np.array([[0,0],[100,0],[80,100],[20,100]])
    np.testing.assert_allclose(marker_centre(corners),[50,62.5])
    assert not np.allclose(marker_centre(corners),corners.mean(axis=0))


def test_generate_printable_markers(tmp_path):
    make_markers(tmp_path)
    assert len(list(tmp_path.glob('marker_*.svg')))==7
    layout=json.loads((tmp_path/'layout.json').read_text())
    assert len(layout['fit'])==4 and len(layout['check'])==3
    assert layout['fit'][0]['floor_m']==[None,None]


def test_parking_preserves_unrelated_config_and_requires_in_region(observations,tmp_path,monkeypatch):
    import perception.calibrate_floor as calibrate
    monkeypatch.setattr(calibrate,'ROOT',tmp_path)
    path=tmp_path/'floor.json';path.write_text(json.dumps(fit(observations)))
    cfg=tmp_path/'robot.yaml'
    cfg.write_text('arm:\n  leader_id: preserve_me\nrealsense:\n  colour: [1280,720]\n  depth: [848,480]\n  fps: 30\n  flip: false\n  align_depth_to_colour: true\nperception:\n  parking_spot_floor_m: [0.22,0]\n')
    with pytest.raises(ValueError,match='outside'): record_parking(path,cfg,[1200,500])
    record_parking(path,cfg,[500,300])
    assert 'leader_id: preserve_me' in cfg.read_text()
    floor=FloorMap.load(path,colour_size=[1280,720],flip=False)
    np.testing.assert_allclose(floor.parking(),[.4,0],atol=1e-6)
    assert len(list((tmp_path/'data/calibration/backups').iterdir()))==2
