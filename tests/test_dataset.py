import pytest
import yaml
from perception.dataset import audit
from perception.listen import summary


def dataset(tmp_path):
    for split in ('train', 'val'):
        for folder in ('images', 'labels'):
            (tmp_path / folder / split).mkdir(parents=True)
        for cls in range(3):
            (tmp_path / 'images' / split / f'{cls}.jpg').write_bytes(f'{split}{cls}'.encode())
            (tmp_path / 'labels' / split / f'{cls}.txt').write_text(f'{cls} .5 .5 .2 .2\n')
    path = tmp_path / 'data.yaml'
    path.write_text(yaml.safe_dump(dict(path=str(tmp_path), train='images/train', val='images/val', names=['phone', 'remote', 'sock'])))
    return path


def test_counts_and_leakage(tmp_path):
    path = dataset(tmp_path)
    assert audit(path)['train']['images_per_class'] == dict(phone=1, remote=1, sock=1)
    (tmp_path / 'images/val/0.jpg').write_bytes(b'train0')
    with pytest.raises(ValueError, match='leakage'):
        audit(path)


def test_bad_box_and_missing_class(tmp_path):
    path = dataset(tmp_path)
    label = tmp_path / 'labels/train/2.txt'
    label.write_text('2 .99 .5 .2 .2')
    with pytest.raises(ValueError, match='normalized'):
        audit(path)
    label.write_text('')
    with pytest.raises(ValueError, match='three classes'):
        audit(path)


def test_other_label_measurements():
    messages = [dict(items=[dict(label='sock', conf=.9, bbox_px=[0, 0, 10, 10])])]
    assert summary(messages, 1, 'left', 'sock')['detected_frames'] == 1
    assert summary(messages, 1, 'left')['detected_frames'] == 0
