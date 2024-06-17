# pocky: A Python bridge to OpenCL

[![build-docs action](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/emprice/pocky/gh-pages/endpoint.json)](https://github.com/emprice/pocky/actions/workflows/docs.yml)
[![License: MIT](https://img.shields.io/github/license/emprice/pocky?style=for-the-badge)](https://opensource.org/licenses/MIT)
![CodeFactor grade](https://img.shields.io/codefactor/grade/github/emprice/pocky/main?logo=codefactor&style=for-the-badge)
![GitHub repo stars](https://img.shields.io/github/stars/emprice/pocky?style=for-the-badge)

## Example usage

```python
import pocky

# Display all OpenCL platforms and devices available
for plat in pocky.list_all_platforms():
    print(plat)
    print(pocky.list_all_devices(plat))
    print()

# Create a context for the default platform
ctx = pocky.Context.default()
```
