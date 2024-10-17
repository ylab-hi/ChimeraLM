use ahash::HashMap;
use ahash::HashMapExt;
use anyhow::Result;
use clap::Parser;
use log::info;
use rayon::prelude::*;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

// 515077e1-bd35-4244-9dbe-55d3a9808d34    0
// 428744c5-9aed-4b3d-84f2-e060f56c2a0b    3       PC3_bulk_WGS_P2_clean.bam,PC3_bulk_WGS_Mk1c_clean.bam,PC3_bulk_WGS_Pacbio_clean.bam
// 150ee2a4-7138-49ab-a75b-575aaba0c09e    3       PC3_bulk_WGS_P2_clean.bam,PC3_bulk_WGS_Mk1c_clean.bam,PC3_bulk_WGS_Pacbio_clean.bam

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    /// path to the support file
    #[arg(value_name = "support file")]
    support_file: PathBuf,

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

fn summary<P: AsRef<Path>>(path: P) -> Result<HashMap<String, usize>> {
    let mut reader = std::io::BufReader::new(std::fs::File::open(path)?);

    let mut line = String::new();
    let mut result = HashMap::new();

    while reader.read_line(&mut line)? > 0 {
        let fields: Vec<&str> = line.trim().split('\t').collect();
        let id = fields[0];
        let count: usize = lexical::parse(fields[1]).unwrap();

        if fields.len() == 2 {
            assert_eq!(count, 0);
        } else {
            assert_eq!(fields.len(), 3);
            let _files = fields[2].split(',').collect::<Vec<&str>>();
        }

        result.insert(id.to_string(), count);
        line.clear();
    }
    Ok(result)
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

    let result = summary(cli.support_file.clone())?;

    // write to json

    let output_prefix = cli.output_prefix.unwrap_or_else(|| {
        cli.support_file
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string()
    });

    let output_file = format!("{}.json", output_prefix);
    let mut writer = std::io::BufWriter::new(std::fs::File::create(output_file)?);
    serde_json::to_writer(writer, &result)?;

    let elapsed = start.elapsed();
    log::info!("elapsed time: {:.2?}", elapsed);
    Ok(())
}
