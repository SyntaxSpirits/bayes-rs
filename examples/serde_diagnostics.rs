#[cfg(feature = "serde")]
use bayes_rs::diagnostics::{McmcDiagnostics, TracePlot};
#[cfg(feature = "serde")]
use nalgebra::DVector;

#[cfg(feature = "serde")]
fn main() -> bayes_rs::Result<()> {
    let samples = vec![
        DVector::from_vec(vec![0.8, 1.7]),
        DVector::from_vec(vec![1.0, 2.0]),
        DVector::from_vec(vec![1.2, 2.3]),
        DVector::from_vec(vec![1.1, 2.1]),
    ];

    let diagnostics = McmcDiagnostics::from_single_chain(&samples)?;
    let trace = TracePlot::new(&samples, 0)?;

    let summary_json = serde_json::to_string_pretty(&diagnostics.summary())
        .expect("diagnostic summary should serialize");
    let trace_json = serde_json::to_string_pretty(&trace).expect("trace plot should serialize");

    println!("diagnostic summary:\n{summary_json}");
    println!("trace plot:\n{trace_json}");

    Ok(())
}

#[cfg(not(feature = "serde"))]
fn main() {
    eprintln!("Run with `cargo run --example serde_diagnostics --features serde` to emit JSON.");
}
