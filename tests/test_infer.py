import pytest

from franks_ml_model import EfficientNetInfer

@pytest.fixture(scope="session")
def model():
    effnet_infer_model = EfficientNetInfer()
    return effnet_infer_model

def test_infer_dog_image_happy_path( model ):
    # model = EfficientNetInfer() # now passed in

    fn_image = 'tests/test_files/dog.jpg'
    top_predictions = model.infer_image( fn_image )

    assert top_predictions[0][0] == 207 # class_id for golden retriever
    assert top_predictions[0][2] > 0.5
    assert len(top_predictions) == 5    # default for topk is 5
    # assert top_predictions[1][0] == 0   # incorrect expectation


def test_infer_non_existent_image_expected_error( model ):
    # model = EfficientNetInfer() # now passed in

    fn_image = 'tests/test_files/not_here.jpg'
    with pytest.raises(FileNotFoundError):
        top_predictions = model.infer_image( fn_image )

def test_effnet_model_forward( model ):
    import torch
    
    effnet_model = model.effnet_model
    image_size = model.img_size
    batch_zero_tensor = torch.zeros(1, 3, image_size, image_size)

    with torch.no_grad():
        outputs = effnet_model( batch_zero_tensor )

    assert outputs.size() == (1, 1000)
    assert outputs.argmax() == 467
