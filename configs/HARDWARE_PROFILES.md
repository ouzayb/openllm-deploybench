# Hardware Profile Examples

Copy the profile that matches your machine to `configs/hardware.local.yaml`, then edit `machine_id`, `provider`, and `hourly_price_usd` as needed.

## Owned / Local

| GPU | GPUs | Config file |
|-----|------|-------------|
| RTX 4090 | 1 | `hardware.owned.rtx4090.single.example.yaml` |
| RTX 4090 | 2 | `hardware.owned.rtx4090.dual.example.yaml` |
| RTX 5090 | 1 | `hardware.owned.rtx5090.single.example.yaml` |
| RTX 5090 | 2 | `hardware.owned.rtx5090.dual.example.yaml` |
| H200 | 1 | `hardware.owned.h200.single.example.yaml` |
| H200 | 2 | `hardware.owned.h200.dual.example.yaml` |

## Cloud / Rented

Set `hourly_price_usd` to your actual billed rate for tokens-per-dollar analysis.

| GPU | GPUs | Config file |
|-----|------|-------------|
| RTX 4090 | 1 | `hardware.cloud.rtx4090.single.example.yaml` |
| RTX 4090 | 2 | `hardware.cloud.rtx4090.dual.example.yaml` |
| RTX 5090 | 1 | `hardware.cloud.rtx5090.single.example.yaml` |
| RTX 5090 | 2 | `hardware.cloud.rtx5090.dual.example.yaml` |
| H200 | 1 | `hardware.cloud.h200.single.example.yaml` |
| H200 | 2 | `hardware.cloud.h200.dual.example.yaml` |

## Quick copy

```bash
# Example: owned single RTX 4090
cp configs/hardware.owned.rtx4090.single.example.yaml configs/hardware.local.yaml

# Example: cloud dual H200
cp configs/hardware.cloud.h200.dual.example.yaml configs/hardware.local.yaml
```
