# Satellite Pass Predictor

SGP4-based satellite pass prediction tool that computes ground station visibility windows over Kourou (ESA spaceport, French Guiana).

## What it does
Takes TLE (orbital element) data for selected satellites, propagates their trajectories over time using the SGP4 model, and calculates when each satellite is visible from Kourou — including elevation, azimuth, and pass duration.

## Status
Early development — currently setting up the project skeleton and data pipeline.

## Tech stack
- Python
- [Skyfield](https://rhodesmill.org/skyfield/) — SGP4 orbit propagation
- Matplotlib — ground track and pass visualization
- TLE data from [Celestrak](https://celestrak.org/)

## Roadmap
- [x] Project setup
- [ ] TLE data ingestion (Celestrak)
- [ ] Orbit propagation + verification against independent source
- [ ] Ground track visualization
- [ ] Visibility window calculator (Kourou)
- [ ] Final output + documentation