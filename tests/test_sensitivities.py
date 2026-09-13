from src.mc_sensitivities.sensitivities import finite_difference_delta


central, forward, backward = finite_difference_delta(
    S=100,
    K=100,
    r=0.05,
    sigma=0.20,
    T=1.0,
    h=1e-4,
    n_paths=100_000,
    seed=42,
)

print("Central Delta:", central)
print("Forward Delta:", forward)
print("Backward Delta:", backward)