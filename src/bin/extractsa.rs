//! Extract chimeric reads from a bam file

use deepbiop::bam::chimeric::{is_chimeric_record, is_retain_record};
use noodles::{bam, bgzf};
use rayon::prelude::*;
use std::fs::File;
use std::num::NonZeroUsize;
use std::thread;
use std::{io, path::Path};

use anyhow::Result;
use log::info;

use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    /// path of bam file
    bam: PathBuf,

    /// maximum number of chimeric reads to extract
    #[arg(short, long)]
    max_reads: Option<usize>,

    /// threads number
    #[arg(short, long, default_value = "2")]
    threads: Option<usize>,

    /// Turn debugging information on
    #[arg(short, long, action = clap::ArgAction::Count)]
    debug: u8,
}

fn select_chimeric_reads<P: AsRef<Path>>(
    bam_path: P,
    max_reads: Option<usize>,
    threads: Option<usize>,
) -> Result<()> {
    let file = File::open(bam_path)?;
    let worker_count = if let Some(threads) = threads {
        NonZeroUsize::new(threads)
            .unwrap()
            .min(thread::available_parallelism().unwrap_or(NonZeroUsize::MIN))
    } else {
        thread::available_parallelism().unwrap_or(NonZeroUsize::MIN)
    };

    let decoder = bgzf::MultithreadedReader::with_worker_count(worker_count, file);
    let mut reader = bam::io::Reader::from(decoder);
    let header = reader.read_header()?;

    let mut res = reader
        .records()
        .par_bridge()
        .filter_map(|result| {
            let record = result.unwrap();
            if is_retain_record(&record) && is_chimeric_record(&record) {
                Some(record)
            } else {
                None
            }
        })
        .collect::<Vec<bam::Record>>();

    info!("total chimeric reads: {}", res.len());

    let stdout = io::stdout().lock();
    let mut writer = bam::io::Writer::new(stdout);

    // only keep max_reads
    res.truncate(max_reads.unwrap_or(res.len()));
    info!("extracted {} chimeric reads", res.len());

    writer.write_header(&header)?;

    for record in res {
        writer.write_record(&header, &record)?;
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

    select_chimeric_reads(cli.bam, cli.max_reads, cli.threads)?;

    let elapsed = start.elapsed();
    log::info!("elapsed time: {:.2?}", elapsed);
    Ok(())
}
