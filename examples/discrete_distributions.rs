use bayes_rs::distributions::{Bernoulli, Binomial, Categorical, DiscreteDistribution, Poisson};
use rand::thread_rng;

fn main() -> bayes_rs::Result<()> {
    let bernoulli = Bernoulli::new(0.3)?;
    let binomial = Binomial::new(10, 0.3)?;
    let poisson = Poisson::new(2.5)?;
    let categorical = Categorical::new(vec![0.2, 0.3, 0.5])?;

    println!("Bernoulli P(X=1) = {:.3}", bernoulli.pmf(1));
    println!("Binomial P(X=3) = {:.3}", binomial.pmf(3));
    println!("Poisson P(X=2) = {:.3}", poisson.pmf(2));
    println!("Categorical P(X=2) = {:.3}", categorical.pmf(2));
    println!(
        "Categorical categories: {}, mean index: {:.3}, variance: {:.3}",
        categorical.category_count(),
        categorical.mean(),
        categorical.variance()
    );

    let mut rng = thread_rng();
    println!("Bernoulli sample: {}", bernoulli.sample(&mut rng));
    println!("Binomial sample: {}", binomial.sample(&mut rng));
    println!("Poisson sample: {}", poisson.sample(&mut rng));
    println!("Categorical sample: {}", categorical.sample(&mut rng));

    Ok(())
}
