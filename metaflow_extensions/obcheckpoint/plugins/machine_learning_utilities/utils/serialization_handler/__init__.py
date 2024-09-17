from .tar import TarHandler

SERIALIZATION_HANDLERS = {
    TarHandler.TYPE: TarHandler,
}
