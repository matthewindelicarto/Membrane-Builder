# Membrane Builder

Build lipid bilayer membranes and predict molecular permeability through them.

## Live Demo

**[Launch Membrane Builder](https://membranebuilder.streamlit.app)** - Free web app, no installation required.

## Features

- Build realistic lipid bilayer membranes with multiple lipid types
- Asymmetric leaflet composition support
- Over 1000 conformers per lipid type for structural diversity
- Predict molecular permeability using physics-based models
- Interactive 3D visualization
- Export to PDB and GRO formats for MD simulations

## Installation

```bash
pip install numpy matplotlib
```

## Quick Start

```bash
python run.py
```

This generates membrane files in the `Outputs/` folder.

## Configuration

Edit `args.txt` to configure your membrane:

```
box_x = 80.0
box_y = 80.0
box_z = 120.0

POPC = 64, 64
CHOL = 32, 32
POPE = 0, 32
```

Format: `LIPID = top_leaflet_count, bottom_leaflet_count`

## Python API

### Building Membranes

```python
from membrane_builder import MembraneBuilder, MembraneConfig

# Simple build
membrane = MembraneBuilder.quick_build(
    lipids={"POPC": (64, 64), "CHOL": (32, 32)},
    box_size=(80, 80)
)
membrane.write_pdb("membrane.pdb")

# Advanced build with templates
config = MembraneConfig.create_simple(
    lipids=[
        {"name": "POPC", "top": 64, "bottom": 64},
        {"name": "CHOL", "top": 32, "bottom": 32}
    ],
    box_size=(80.0, 80.0, 120.0)
)

builder = MembraneBuilder(seed=12345)
membrane = builder.build(config, use_templates=True, templates_dir="Lipids")
membrane.write_pdb("membrane.pdb")
membrane.write_gro("membrane.gro")
```

### Predicting Permeability

```python
from membrane_builder import MoleculeDescriptor, PermeabilityPredictor

# Create molecule descriptor
caffeine = MoleculeDescriptor.simple(
    name="caffeine",
    molecular_weight=194.2,
    total_asa=150.0,
    n_hbd=0,
    n_hba=3,
    charge=0
)

# Calculate permeability
predictor = PermeabilityPredictor()
result = predictor.calculate(caffeine, membrane_type="POPC")

print(f"log P = {result.log_p:.2f}")
print(f"Permeability = {result.permeability:.2e} cm/s")
```

### Preset Molecules

```python
from membrane_builder import MoleculeDescriptor

water = MoleculeDescriptor.simple("water", mw=18, asa=40, hbd=2, hba=1)
ethanol = MoleculeDescriptor.simple("ethanol", mw=46, asa=80, hbd=1, hba=1)
glucose = MoleculeDescriptor.simple("glucose", mw=180, asa=180, hbd=5, hba=6)
caffeine = MoleculeDescriptor.simple("caffeine", mw=194, asa=150, hbd=0, hba=3)
aspirin = MoleculeDescriptor.simple("aspirin", mw=180, asa=140, hbd=1, hba=4, charge=-1, pka=3.5)
```

## Available Lipids

| Name | Type | Charge | Description |
|------|------|--------|-------------|
| POPC | PC | 0 | Phosphatidylcholine - standard membrane lipid |
| DOPC | PC | 0 | Dioleoylphosphatidylcholine - unsaturated, fluid |
| DPPC | PC | 0 | Dipalmitoylphosphatidylcholine - saturated, rigid |
| POPE | PE | 0 | Phosphatidylethanolamine - bacterial membranes |
| POPG | PG | -1 | Phosphatidylglycerol - anionic, bacterial |
| POPS | PS | -1 | Phosphatidylserine - anionic, inner leaflet |
| CHOL | Sterol | 0 | Cholesterol - stiffens membrane |
| PSM | SM | 0 | Sphingomyelin - lipid rafts |

### Adding New Lipids

1. Create folder: `Lipids/newlipid/`
2. Add conformer subfolders: `Lipids/newlipid/conf1/`, `conf2/`, etc.
3. Place `.crd` or `.pdb` files in each conformer folder

## Membrane Architecture

- Lipids placed on hexagonal grid for optimal packing
- Top leaflet: headgroups face +Z
- Bottom leaflet: headgroups face -Z
- Phosphorus atoms at ±18 Å from center

## Permeability Model

The model calculates transfer free energy at each position through the membrane based on:
- Hydrogen bond donors/acceptors
- Molecular surface area
- Charge state
- Membrane composition

| Component | Effect |
|-----------|--------|
| Cholesterol | Reduces permeability |
| Unsaturated lipids | Increases permeability |
| Sphingomyelin | Reduces permeability |

### Classification

- **High**: log P > -6
- **Moderate**: -6 > log P > -8
- **Low**: log P < -8

## Output Files

- `membrane.pdb` - Structure file (VMD, PyMOL, Chimera compatible)
- `membrane.gro` - GROMACS format
- `membrane_report.txt` - Summary

## References

1. Lomize AL, Pogozheva ID. Physics-based method for modeling passive membrane permeability. J Chem Inf Model. 2019.

2. Nagle JF, Tristram-Nagle S. Structure of lipid bilayers. Biochim Biophys Acta. 2000.

3. Jo S, et al. CHARMM-GUI. J Comput Chem. 2008.

## License

MIT License
