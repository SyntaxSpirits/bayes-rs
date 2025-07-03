use bayes_rs::distributions::{Distribution, MultivariateDistribution, MultivariateNormal, Normal};
use bayes_rs::samplers::{HamiltonianMonteCarlo, MetropolisHastings, Sampler};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nalgebra::{DMatrix, DVector};

fn bench_metropolis_hastings(c: &mut Criterion) {
    c.bench_function("metropolis_hastings_1d", |b| {
        let log_posterior = |params: &DVector<f64>| -> f64 {
            let normal = Normal::new(0.0, 1.0).unwrap();
            normal.log_pdf(params[0])
        };

        let initial_state = DVector::from_vec(vec![0.0]);
        let proposal_std = DVector::from_vec(vec![0.5]);

        b.iter(|| {
            let mut sampler = MetropolisHastings::new(
                black_box(&log_posterior),
                black_box(initial_state.clone()),
                black_box(proposal_std.clone()),
            )
            .unwrap();

            black_box(sampler.sample(1000))
        })
    });

    c.bench_function("metropolis_hastings_10d", |b| {
        let log_posterior = |params: &DVector<f64>| -> f64 {
            let mu = DVector::zeros(10);
            let cov = DMatrix::identity(10, 10);
            let mvn = MultivariateNormal::new(mu, cov).unwrap();
            mvn.log_pdf(params)
        };

        let initial_state = DVector::zeros(10);
        let proposal_std = DVector::from_element(10, 0.5);

        b.iter(|| {
            let mut sampler = MetropolisHastings::new(
                black_box(&log_posterior),
                black_box(initial_state.clone()),
                black_box(proposal_std.clone()),
            )
            .unwrap();

            black_box(sampler.sample(1000))
        })
    });
}

fn bench_hamiltonian_monte_carlo(c: &mut Criterion) {
    c.bench_function("hmc_1d", |b| {
        let log_posterior = |params: &DVector<f64>| -> f64 {
            let normal = Normal::new(0.0, 1.0).unwrap();
            normal.log_pdf(params[0])
        };

        let gradient = |params: &DVector<f64>| -> DVector<f64> {
            // Gradient of log N(0,1) = -x
            DVector::from_vec(vec![-params[0]])
        };

        let initial_state = DVector::from_vec(vec![0.0]);
        let step_size = 0.1;
        let n_leapfrog = 10;

        b.iter(|| {
            let mut sampler = HamiltonianMonteCarlo::new(
                black_box(&log_posterior),
                black_box(&gradient),
                black_box(initial_state.clone()),
                black_box(step_size),
                black_box(n_leapfrog),
            )
            .unwrap();

            black_box(sampler.sample(1000))
        })
    });

    c.bench_function("hmc_10d", |b| {
        let log_posterior = |params: &DVector<f64>| -> f64 {
            let mu = DVector::zeros(10);
            let cov = DMatrix::identity(10, 10);
            let mvn = MultivariateNormal::new(mu, cov).unwrap();
            mvn.log_pdf(params)
        };

        let gradient = |params: &DVector<f64>| -> DVector<f64> {
            // Gradient of log MVN with identity covariance = -x
            -params
        };

        let initial_state = DVector::zeros(10);
        let step_size = 0.1;
        let n_leapfrog = 10;

        b.iter(|| {
            let mut sampler = HamiltonianMonteCarlo::new(
                black_box(&log_posterior),
                black_box(&gradient),
                black_box(initial_state.clone()),
                black_box(step_size),
                black_box(n_leapfrog),
            )
            .unwrap();

            black_box(sampler.sample(1000))
        })
    });
}

fn bench_single_step(c: &mut Criterion) {
    c.bench_function("mh_single_step", |b| {
        let log_posterior = |params: &DVector<f64>| -> f64 {
            let normal = Normal::new(0.0, 1.0).unwrap();
            normal.log_pdf(params[0])
        };

        let initial_state = DVector::from_vec(vec![0.0]);
        let proposal_std = DVector::from_vec(vec![0.5]);

        let mut sampler =
            MetropolisHastings::new(&log_posterior, initial_state, proposal_std).unwrap();

        b.iter(|| black_box(sampler.step()))
    });

    c.bench_function("hmc_single_step", |b| {
        let log_posterior = |params: &DVector<f64>| -> f64 {
            let normal = Normal::new(0.0, 1.0).unwrap();
            normal.log_pdf(params[0])
        };

        let gradient =
            |params: &DVector<f64>| -> DVector<f64> { DVector::from_vec(vec![-params[0]]) };

        let initial_state = DVector::from_vec(vec![0.0]);
        let step_size = 0.1;
        let n_leapfrog = 10;

        let mut sampler = HamiltonianMonteCarlo::new(
            &log_posterior,
            &gradient,
            initial_state,
            step_size,
            n_leapfrog,
        )
        .unwrap();

        b.iter(|| black_box(sampler.step()))
    });
}

fn bench_distributions(c: &mut Criterion) {
    c.bench_function("normal_log_pdf", |b| {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let x = 1.0;

        b.iter(|| black_box(normal.log_pdf(black_box(x))))
    });

    c.bench_function("mvn_log_pdf_10d", |b| {
        let mu = DVector::zeros(10);
        let cov = DMatrix::identity(10, 10);
        let mvn = MultivariateNormal::new(mu, cov).unwrap();
        let x = DVector::from_element(10, 0.5);

        b.iter(|| black_box(mvn.log_pdf(black_box(&x))))
    });
}

criterion_group!(
    benches,
    bench_metropolis_hastings,
    bench_hamiltonian_monte_carlo,
    bench_single_step,
    bench_distributions
);
criterion_main!(benches);
