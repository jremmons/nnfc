# HACK need to import torch to ensure that the all the required
# dynamic libraries are loaded before importing nnfc_codec. We set the
# module to `None` so that it cannot be used after import.
import torch as __nnfc_torch; __nnfc_torch = None

# try to import the cuda functions into the global namespace
try:
    from ._ext.nnfc_codec import tensor_memcpy_d2h
    from ._ext.nnfc_codec import tensor_memcpy_h2d
except:
    # could not import the cuda functions
    pass

from ._ext.nnfc_codec import available_decoders
from ._ext.nnfc_codec import available_encoders
