# OutflowBC_KRATOS
Improved Outflow Boundary Conditions for Fluid Dynamics in KRATOS. 

Input block example in ProjectParameters.json for Convective outlet condition:

```json

{
        "python_module" : "impose_convective_outlet_process",
        "Parameters"    : {
            "fluid_model_part_name" :  "FluidModelPart.Parts_fluid",
            "outlet_model_part_name" : "FluidModelPart.Outlet2D_outlet",
            "print_to_screen": true,
            "P_inf"  :0.00,
            "L_inf"  :1800,
            "U_ref"  : 25.0
        }
}

```

Input block example in ProjectParameters.json for Sponge Layer condition:

```json
{
    "python_module" : "impose_sponge_layer_process",
    "Parameters"    : {
        "fluid_model_part_name" : "FluidModelPart.Parts_Fluid",
        "print_to_screen": true,
            "x_start" : 1000,
            "x_end"   : 1300,
            "y_start" : -450,
            "y_end"   : 450,
            "z_start" : 0.0,
            "z_end"   : 600.0,
        "damping_coefficent"   : 1.00,
        "profile_type"    : "ud_exponential",
        "u_mean"           : 40.0,
        "z_ref"           : 180,
        "alpha"           : 0.25
    }
}
```
References:
Khallouf, A. (2021). Developments for Improved FEM-Based LES Simulations in Computational Wind Engineering. (Master's Thesis).Technical University of Muncih. 
