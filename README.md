# Metavoice Truss
## Notes
- We are mostly following the server.py convention from the metavoice repo
- We reference a fork of the metavoice repo that includes changes that lets us treat the repo as a package and install via Truss's standard config requirements
- flash_attn seems to require installation with `--no-build-isolation`. As this isn't supported, installing the wheel directly seems to work
- Invoke with `truss predict -d '{"text": "This is a metavoice test"}' | python process.py` (as per https://github.com/basetenlabs/truss-examples/tree/main/bark)

## To Dos
- Add support for picking the reference voice to use (e.g. see Whisper Truss example)
- Clean up config management (right now mostly copy-pasted from server.py)
- Accept longer text and split it to fit within the model limit
- Cache weights
- Move out process.py to a separate file (or have model upload the file to a remote server)
- Get model to run on A10
