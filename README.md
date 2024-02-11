# Metavoice Truss
## Notes
- We are mostly following the server.py convention from the metavoice repo
- We reference a fork of the metavoice repo that includes changes that lets us treat the repo as a package and install via Truss's standard config requirements
- flash_attn seems to require installation with `--no-build-isolation`. As this isn't supported, installing the wheel directly seems to work