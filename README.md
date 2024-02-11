# Metavoice Truss
Truss for [Metavoice](https://github.com/metavoiceio/metavoice-src).

## Notes
- Follows the server.py convention from the metavoice repo
- Original metavoice repo is not fully configured as a library, so we use a fork that's been updated so that it can be included as a standard requirement in the `config.yaml` file
- `flash_attn` requires installation with `--no-build-isolation`. As this isn't supported, installing the wheel directly seems to work.
- Invoke with `truss predict -d '{"text": "This is a metavoice test"}' | python process.py` (similar to [Bark Truss example](https://github.com/basetenlabs/truss-examples/tree/main/bark)) – be aware of current 220 character limit on Metavoice

## To Dos
- Add support for picking the reference voice to use (e.g. see Whisper Truss example)
- Clean up config management (right now mostly copy-pasted from server.py)
- Accept longer text and split it to fit within the model limit
- Cache weights
- Move out process.py to a separate directory (or have model upload the file to a remote server)
- Get model to run on A10
