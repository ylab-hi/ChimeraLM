use ahash::HashSet;
use anyhow::Result;
use bstr::BString;
use clap::Parser;
use log::info;
use rayon::prelude::*;
use std::{
    io::BufRead,
    path::{Path, PathBuf},
};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    /// path to the fq
    fq: PathBuf,

    /// path to positive reads
    pos: PathBuf,

    /// path to negative reads
    neg: PathBuf,

    /// threads number
    #[arg(short, long, default_value = "2")]
    threads: Option<usize>,

    /// Turn debugging information on
    #[arg(short, long, action = clap::ArgAction::Count)]
    debug: u8,
}

fn load_reads<P: AsRef<Path>>(path: P) -> Result<HashSet<BString>> {
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let reads = reader
        .lines()
        .map(|l| l.unwrap().split_whitespace().next().unwrap().into())
        .collect();
    Ok(reads)
}

fn worker<P: AsRef<Path>>(fq: P, pos: P, neg: P, threads: Option<usize>) -> Result<()> {
    let pos_reads = load_reads(pos)?;
    info!("pos reads: {}", pos_reads.len());

    let neg_reads = load_reads(neg)?;
    info!("neg reads: {}", neg_reads.len());

    let mut records = deepbiop::fastq::io::read_noodle_records(&fq)?;

    info!("add target for records: {}", records.len());

    // if record name is in pos_reads, set target to 1, then 0
    // add read name with 1 for example, a|1, b|0
    records.par_iter_mut().for_each(|record| {
        let id: BString = record.name().into();
        if pos_reads.contains(&id) {
            *record.name_mut() = BString::from(format!("{}|1", id));
        } else if neg_reads.contains(&id) {
            *record.name_mut() = BString::from(format!("{}|0", id));
        }
    });

    let result_path = format!(
        "{}.target.fq.gz",
        fq.as_ref().file_stem().unwrap().to_string_lossy()
    );

    info!("write to {}", &result_path);

    deepbiop::fastq::io::write_bgzip_fq_parallel_for_noodle_record(
        &records,
        result_path.into(),
        threads,
    )?;
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
    worker(&cli.fq, &cli.pos, &cli.neg, cli.threads)?;

    let elapsed = start.elapsed();
    log::info!("elapsed time: {:.2?}", elapsed);

    Ok(())
}
