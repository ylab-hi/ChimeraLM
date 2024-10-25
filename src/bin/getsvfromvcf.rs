use ahash::HashMap;
use anyhow::Result;
use clap::Parser;
use log::info;
use rayon::prelude::*;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    /// path to the vcf
    vcf: PathBuf,

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

//GL000225.1      95637   cuteSV.DEL.76   GATGTCACTTTTGTCAAGGATATGGCTACAGGGACATTGTGACATGTAAATGCACGATCACACATCT       G       0.0     q5      IMPRECISE;SVTYPE=DEL;SVLEN=-66;END=95703;CIPOS=-0,0;CILEN=-0,0;RE=4;RNAMES=c21b6509-04d2-4fc9-ba53-9ad2c86a72fa,660e9194-477c-491e-9cb5-f0a84aff44a9,34dd3621-66ae-41cb-a8c9-5526ced04cb3,66ed5297-bde8-43df-8553-5693d36eda31;AF=0.1739;STRAND=+-  GT:DR:DV:PL:GQ    0/0:19:4:0,21,143:20

fn find_needle<'a>(needle: &'a str, haystack: Vec<&'a str>) -> &'a str {
    haystack
        .par_iter()
        .find_first(|&&x| x.starts_with(needle))
        .unwrap()
}

fn worker<P: AsRef<Path>>(vcf_path: P) -> Result<()> {
    // Implement your worker logic here
    let file = std::fs::File::open(vcf_path.as_ref()).unwrap();
    let mut buffer_reader = std::io::BufReader::new(file);

    let mut line = String::new();

    while buffer_reader.read_line(&mut line)? > 0 {
        // igore comment lines
        if line.starts_with("#") {
            line.clear();
            continue;
        }
        let columns = line.split('\t').collect::<Vec<&str>>();
        // get info field
        let info_field = columns[7];
        // get RNAME field
        let infos = info_field.split(';').collect::<Vec<&str>>();

        let rname = find_needle("RNAMES", infos);

        println!("rname_field: {:?}", rname);
        line.clear();
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
    worker(&cli.vcf)?;

    let elapsed = start.elapsed();
    log::info!("elapsed time: {:.2?}", elapsed);

    Ok(())
}
