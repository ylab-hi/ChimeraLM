use ahash::HashMap;
use ahash::HashMapExt;
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
fn find_needle<'a>(needle: &'a str, haystack: &[&'a str]) -> &'a str {
    haystack
        .par_iter()
        .find_first(|&&x| x.starts_with(needle))
        .unwrap()
}
// RNAMES=8c43bf15-f889-4fab-8af1-489da9179818,c241a190-48de-4756-9c10-c22153de49
// 0d,e25bcdd7-6cea-4275-ae3a-4e638cf56b3e,b7c55f70-6559-463c-94a5-0e236b1aa803,ecbf62d0-031e-4
// f0a-950d-1bf701b9b9d5,0c7e2d6e-e9af-4128-8388-aabe7a6a736d,2f308d2f-c026-458b-b38f-cec1a52c4
// 38d,45467d9b-bf8c-4d8c-a4f7-ff2db1c062ca,89f69efe-a605-449d-837e-f91456207ab0,bbebb159-d93e-
// 4756-a59f-cc9162b46c78,e7b30e06-d847-4dbd-ab9a-ef3423742552,d2b6f6ef-819c-4e3d-b075-422ee2ca
// 7ca4
fn parse_rname(rname: &str) -> Vec<String> {
    let rname = rname.strip_prefix("RNAMES=").unwrap();
    rname
        .split(',')
        .map(|s| s.to_string())
        .collect::<Vec<String>>()
}

fn write_results(results: &HashMap<String, Vec<String>>, file_name: &str) -> Result<()> {
    let suffix = format!("{}.sv.read.sup.txt", file_name);
    let output_path = suffix;
    let mut buf_writer = std::io::BufWriter::new(std::fs::File::create(output_path)?);

    for (svtype, reads) in results.iter() {
        for read in reads.iter() {
            let line = format!("{}\t{}\n", read, svtype);
            buf_writer.write_all(line.as_bytes())?;
        }
    }
    Ok(())
}

fn worker<P: AsRef<Path>>(vcf_path: P) -> Result<()> {
    // Implement your worker logic here
    let file = std::fs::File::open(vcf_path.as_ref()).unwrap();
    let mut buffer_reader = std::io::BufReader::new(file);

    let mut line = String::new();

    let mut result = HashMap::new();

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

        let rname = find_needle("RNAMES", &infos);
        let read_names = parse_rname(rname);

        let svtype_str = find_needle("SVTYPE", &infos);
        let svtype = svtype_str.strip_prefix("SVTYPE=").unwrap();

        result.insert(svtype.to_string(), read_names);
        line.clear();
    }

    write_results(&result, vcf_path.as_ref().to_str().unwrap())?;
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
