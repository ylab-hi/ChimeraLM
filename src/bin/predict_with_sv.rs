use ahash::HashMap;
use ahash::HashMapExt;

use anyhow::Result;
use clap::Parser;
use log::info;
use rayon::prelude::*;
use std::io::BufRead;
use std::io::Write;
use std::path::{Path, PathBuf};

use chimera::predict::load_predicts_from_batch_pts;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    /// path to the predict
    predict: PathBuf,

    /// path to the sv
    sv: PathBuf,

    /// max predicts
    #[arg(short, long)]
    max_predicts: Option<usize>,

    /// threads number
    #[arg(short, long, default_value = "2")]
    threads: Option<usize>,

    /// Turn debugging information on
    #[arg(short, long, action = clap::ArgAction::Count)]
    debug: u8,
}

fn load_sv<P: AsRef<Path>>(path: P) -> HashMap<String, String> {
    let file = std::fs::File::open(path).unwrap();
    let reader = std::io::BufReader::new(file);
    let mut svs = HashMap::new();

    for line in reader.lines() {
        let line = line.unwrap();
        let mut iter = line.split_whitespace();
        let read_name = iter.next().unwrap();
        let sv_type = iter.next().unwrap();
        svs.insert(read_name.to_string(), sv_type.to_string());
    }
    svs
}

fn worker<P: AsRef<Path>>(predict_path: P, sv_path: P, max_predict: Option<usize>) -> Result<()> {
    let predicts = load_predicts_from_batch_pts(predict_path.as_ref().to_path_buf(), max_predict)?;
    info!("predicts: {:?}", predicts.len());

    let svs = load_sv(sv_path);
    info!("svs: {:?}", svs.len());

    // write predicts to stdout
    let mut writer = std::io::BufWriter::new(std::io::stdout());
    for (id, predict) in predicts {
        let na = "NA".to_string();
        let sv = svs.get(&id).unwrap_or(&na);
        writeln!(writer, "{},{},{}", id, predict.label, sv)?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let start = std::time::Instant::now();
    let cli = Cli::parse();

    let log_level = match cli.debug {
        0 => log::LevelFilter::Info,
        1 => log::LevelFilter::Debug,
        2 => log::LevelFilter::Trace,
        _ => log::LevelFilter::Trace,
    };
    // set log level
    env_logger::builder().filter_level(log_level).init();

    rayon::ThreadPoolBuilder::new()
        .num_threads(cli.threads.unwrap())
        .build_global()
        .unwrap();

    info!("{:?}", cli);
    worker(&cli.predict, &cli.sv, cli.max_predicts)?;

    let elapsed = start.elapsed();
    log::info!("elapsed time: {:.2?}", elapsed);

    Ok(())
}
