//! ZipEnhancer Rust - Audio Denoise Processor

use clap::Parser;
use std::process;
use std::path::Path;
use zipenhancer::{init_logging, Args, Result};

fn main() {
    let args = Args::parse();

    if let Err(e) = check_and_setup_onnx_library(&args) {
        eprintln!("Error: {}", e);
        process::exit(1);
    }

    init_logging(args.verbose);

    if let Err(e) = run(args) {
        eprintln!("Error: {}", e);
        process::exit(1);
    }
}

fn run(args: Args) -> Result<()> {
    if args.verbose {
        println!("{}", zipenhancer::get_library_info());
        println!();
    }

    if args.test_only {
        return run_test_mode();
    }

    let config = zipenhancer::config::Config::from_args_and_config(args.clone())?;

    if !config.input_path.exists() {
        return Err(zipenhancer::ZipEnhancerError::config(format!(
            "Input file does not exist: {}", config.input_path.display()
        )));
    }

    if !config.model_path().exists() {
        return Err(zipenhancer::ZipEnhancerError::config(format!(
            "Model file does not exist: {}", config.model_path().display()
        )));
    }

    if let Some(parent) = config.output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    println!("=== ZipEnhancer Audio Denoise Processor ===");
    println!("Input: {}", config.input_path.display());
    println!("Output: {}", config.output_path.display());

    if args.serial {
        // Serial mode: single session with multi-threading
        println!("Mode: Serial ({} threads)", config.inference_threads());
        println!("============================================\n");

        let mut processor = zipenhancer::processing::AudioProcessor::new(config.clone())?;
        processor.warm_up()?;
        let result = processor.process_file(&config.input_path, &config.output_path)?;

        println!("=== Processing Complete ===");
        println!("Time: {:.2}s", result.performance_metrics.processing_time_seconds);
        println!("RTF: {:.3}", result.performance_metrics.real_time_factor);
        if config.verbose() {
            println!("Segments: {}", result.performance_metrics.segment_count);
            println!("Avg inference: {:.2}ms", result.performance_metrics.average_inference_time_ms);
        }
    } else {
        // Parallel mode: multiple sessions
        let num_workers = args.parallel_workers.unwrap_or(4);
        println!("Mode: Parallel ({} workers)", num_workers);
        println!("============================================\n");

        let mut processor = zipenhancer::processing::ParallelAudioProcessor::new(config.clone(), num_workers)?;
        let result = processor.process_file(&config.input_path, &config.output_path)?;

        println!("=== Processing Complete ===");
        println!("Time: {:.2}s", result.processing_time_secs);
        println!("RTF: {:.3}", result.rtf);
        if config.verbose() {
            println!("Segments: {}", result.segment_count);
            println!("Workers: {}", result.worker_count);
            println!("Avg inference: {:.2}ms", result.avg_inference_time_ms);
        }
    }

    Ok(())
}

fn run_test_mode() -> Result<()> {
    println!("=== Test Mode ===");
    let config = zipenhancer::config::Config::default();
    config.validate()?;
    println!("✅ Config OK");
    println!("✅ CPU cores: {}", zipenhancer::config::utils::cpu_count());
    println!("✅ Ready for processing");
    Ok(())
}

fn check_and_setup_onnx_library(args: &Args) -> Result<()> {
    if let Some(ref lib_path) = args.onnx_lib {
        if !lib_path.exists() {
            return Err(zipenhancer::ZipEnhancerError::config(format!(
                "ONNX Runtime library not found: {}", lib_path.display()
            )));
        }
        unsafe {
            std::env::set_var("LD_LIBRARY_PATH", format!("{}:{}",
                lib_path.parent().unwrap_or(Path::new(".")).display(),
                std::env::var("LD_LIBRARY_PATH").unwrap_or_default()
            ));
        }
        if args.verbose { println!("Using ONNX Runtime: {}", lib_path.display()); }
        return Ok(());
    }

    let paths = [
        "/opt/homebrew/lib/libonnxruntime.dylib",
        "/usr/local/lib/libonnxruntime.dylib",
        "lib/libonnxruntime.1.24.0.dylib",
        "lib/onnxruntime_sdk/lib/libonnxruntime.dylib",
    ];

    for path in &paths {
        if Path::new(path).exists() {
            unsafe {
                std::env::set_var("LD_LIBRARY_PATH", format!("{}:{}",
                    Path::new(path).parent().unwrap_or(Path::new(".")).display(),
                    std::env::var("LD_LIBRARY_PATH").unwrap_or_default()
                ));
            }
            if args.verbose { println!("Found ONNX Runtime: {}", path); }
            return Ok(());
        }
    }

    Err(zipenhancer::ZipEnhancerError::config("ONNX Runtime not found. Use --onnx-lib to specify path."))
}
