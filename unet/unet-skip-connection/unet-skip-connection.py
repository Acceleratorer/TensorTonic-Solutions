import numpy as np

def crop_and_concat(encoder_features: np.ndarray, decoder_features: np.ndarray) -> np.ndarray:
    """
    Crop encoder features and concatenate with decoder features.
    Assumes NHWC: (batch, height, width, channels)
    """
    _, h_enc, w_enc, _ = encoder_features.shape
    _, h_dec, w_dec, _ = decoder_features.shape

    crop_h = h_enc - h_dec
    crop_w = w_enc - w_dec

    start_h = crop_h // 2
    start_w = crop_w // 2
    end_h = start_h + h_dec
    end_w = start_w + w_dec

    encoder_cropped = encoder_features[:, start_h:end_h, start_w:end_w, :]
    return np.concatenate([encoder_cropped, decoder_features], axis=-1)