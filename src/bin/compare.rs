use anyhow::Result;
use clap::Parser;
use log::debug;
use log::info;
use std::io::BufRead;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    /// path to the chimeric file
    #[arg(value_name = "chimeric file")]
    chimeric_file: PathBuf,

    /// threads number
    #[arg(short, long, default_value = "2")]
    threads: Option<usize>,

    /// prefix for output files
    #[arg(short, long)]
    output_prefix: Option<String>,

    /// Turn debugging information on
    #[arg(short, long, action = clap::ArgAction::Count)]
    debug: u8,
}

fn worker<P: AsRef<Path>>(path: P) -> Result<Vec<deepbiop::bam::chimeric::ChimericEvent>> {
    let mut reader = std::io::BufReader::new(std::fs::File::open(path.as_ref())?);

    let mut res = vec![];

    let mut line = String::new();
    while reader.read_line(&mut line)? > 0 {
        debug!("parse {}", line);
        let fields = line.trim().split('\t').collect::<Vec<&str>>();
        assert_eq!(fields.len(), 3);
        let event = deepbiop::bam::chimeric::ChimericEvent::parse_list_pos(fields[2], fields[0])?;
        res.push(event);
        line.clear();
    }

    info!(
        "collect {} events from {}",
        res.len(),
        path.as_ref().display()
    );
    Ok(res)
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
    worker(cli.chimeric_file)?;

    let elapsed = start.elapsed();
    log::info!("elapsed time: {:.2?}", elapsed);
    Ok(())
}
