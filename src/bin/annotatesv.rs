//! Annotate structral variant events in dirty bam with clean bam
//!
//! The output is a table with the following columns:

use ahash::HashMap;
use ahash::HashMapExt;

use anyhow::Result;
use clap::Parser;
use deepbiop::utils::sv;
use log::debug;
use log::info;
use rayon::prelude::*;
use std::io::BufRead;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::str::FromStr;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    /// path to the clean vcf
    #[arg(long="cvcf", value_name = "clean vcf", action = clap::ArgAction::Append)]
    cvcfs: Vec<PathBuf>,

    /// path to dirty vcf
    #[arg(long="dvcf", value_name = "dirty vcf", action = clap::ArgAction::Append)]
    dvcfs: Vec<PathBuf>,

    // threshold for comparing two svs
    #[arg(long, default_value = "1000")]
    threshold: usize,

    /// threads number
    #[arg(short, long, default_value = "2")]
    threads: Option<usize>,

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

// KI270718.1      38055   cuteSV.BND.125  N       N[chr17:48407264[       20.3
// PASS    SVTYPE=TRA;END=48407264;SVLEN=.;CHR2=chr17;SUPPORT=2;SVMETHOD=octopusV;RTID=.;AF=1.0;STRAND=.;
// RNAMES=0fff4336-8d6c-4166-962b-6e920321db65,fd258b9c-780b-4403-a1f9-475f195bc107
// GT:AD:LN:ST:QV:TY:ID:SC:REF:ALT:CO
// 1/1:0,2:.:.:20.3:TRA:cuteSV.BND.125:cuteSV-2.1.1:N:N[chr17:48407264[:KI270718.1_38055-chr17_48407264

fn get_sv_from_vcf<P: AsRef<Path>>(
    vcf_path: P,
) -> Result<Vec<(sv::StructuralVariant, Vec<String>)>> {
    // Implement your worker logic here
    let file = std::fs::File::open(vcf_path.as_ref()).unwrap();
    let mut buffer_reader = std::io::BufReader::new(file);

    let mut line = String::new();
    let mut result = vec![];

    while buffer_reader.read_line(&mut line)? > 0 {
        // ignore comment lines
        if line.starts_with("#") {
            line.clear();
            continue;
        }
        debug!("line: {}", line);

        let columns = line.split('\t').collect::<Vec<&str>>();

        // get config field
        let chr = columns[0];
        let breakpoint1 = columns[1].parse::<usize>().unwrap();

        // get info field
        let info_field = columns[7];
        // get RNAME field
        let infos = info_field.split(';').collect::<Vec<&str>>();

        let rname = find_needle("RNAMES", &infos);
        let read_names = parse_rname(rname);

        let svtype_str = find_needle("SVTYPE", &infos);
        let svtype = svtype_str.strip_prefix("SVTYPE=").unwrap();

        let breakpoint2_str = find_needle("END", &infos);
        let breakpoint2 = breakpoint2_str
            .strip_prefix("END=")
            .unwrap()
            .parse::<usize>()
            .unwrap();

        let svtype = sv::StructralVariantType::from_str(svtype)?;

        let sv = sv::StructuralVariantBuilder::default()
            .sv_type(svtype)
            .chr(chr.into())
            .breakpoint1(breakpoint1)
            .breakpoint2(breakpoint2)
            .build()
            .unwrap();

        result.push((sv, read_names));
        line.clear();
    }

    info!("read {} svs", result.len());

    Ok(result)
}

fn compare_sv(sv1: &sv::StructuralVariant, sv2: &sv::StructuralVariant, threshold: usize) -> bool {
    if sv1.chr != sv2.chr {
        return false;
    }

    if sv1.sv_type != sv2.sv_type {
        return false;
    }

    if sv1.breakpoint1.abs_diff(sv2.breakpoint2).le(&threshold) {
        return true;
    }

    if sv1.breakpoint2.abs_diff(sv2.breakpoint2).le(&threshold) {
        return true;
    }

    false
}

fn write_result<P: AsRef<Path>>(
    output_prefix: P,
    result: &[(PathBuf, sv::StructuralVariant, String, String)],
) -> Result<()> {
    let output_path = output_prefix.as_ref().with_extension("tsv");
    let mut writer = std::fs::File::create(output_path)?;

    writeln!(
        writer,
        "dirty_sv\tdirty_sv_type\tclean_sv\tmatched_read_names_in_dirty"
    )?;

    for (dirty_sv, dirty_sv_type, clean_sv, matched_read_names_in_dirty) in result {
        writeln!(
            writer,
            "{}\t{:?}\t{}\t{}",
            dirty_sv.to_str().unwrap(),
            dirty_sv_type,
            clean_sv,
            matched_read_names_in_dirty
        )?;
    }

    Ok(())
}

fn worker(cvcfs: &[PathBuf], dvcfs: &[PathBuf], threshold: usize) -> Result<()> {
    // Load clean SVs with error handling
    let clean_svs: Result<HashMap<PathBuf, Vec<(sv::StructuralVariant, Vec<String>)>>> = cvcfs
        .par_iter()
        .map(|cvcf| get_sv_from_vcf(cvcf).map(|svs| (cvcf.clone(), svs)))
        .collect();
    let clean_svs = clean_svs?;

    // Load dirty SVs with error handling
    let dirty_svs: Result<HashMap<PathBuf, Vec<(sv::StructuralVariant, Vec<String>)>>> = dvcfs
        .par_iter()
        .map(|dvcf| get_sv_from_vcf(dvcf).map(|svs| (dvcf.clone(), svs)))
        .collect();
    let dirty_svs = dirty_svs?;

    // Pre-allocate result HashMap with capacity
    let mut result = HashMap::with_capacity(dirty_svs.len());

    // Process each dirty SV file
    for (dvcf, dirty_svs) in dirty_svs {
        let infos = dirty_svs
            .par_iter()
            .map(|(dirty_sv, dirty_read_names)| {
                // Find matching clean SVs
                let matched_cvcfs: Vec<_> = clean_svs
                    .par_iter()
                    .filter_map(|(cvcf, clean_svs)| {
                        // Use any_parallel for better performance on large lists
                        if clean_svs
                            .par_iter()
                            .any(|(clean_sv, _)| compare_sv(clean_sv, dirty_sv, threshold))
                        {
                            Some(cvcf)
                        } else {
                            None
                        }
                    })
                    .collect();

                // Join results with error handling
                let matched_cvcfs = matched_cvcfs
                    .iter()
                    .filter_map(|s| s.to_str())
                    .collect::<Vec<_>>()
                    .join(",");

                let joined_dirty_read_names = dirty_read_names.join(",");

                (
                    dvcf.clone(),
                    dirty_sv.clone(),
                    matched_cvcfs,
                    joined_dirty_read_names,
                )
            })
            .collect::<Vec<_>>();

        result.insert(dvcf, infos);
    }

    // Write results
    for (vcf_path, results) in result {
        let stem = vcf_path
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| anyhow::anyhow!("Invalid file name"))?;

        let output_prefix = format!("{}_annotated_sv", stem);
        info!("Writing annotated SV result to {}.tsv", output_prefix);
        write_result(&output_prefix, &results)?;
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

    worker(&cli.cvcfs, &cli.dvcfs, cli.threshold)?;

    let elapsed = start.elapsed();
    log::info!("elapsed time: {:.2?}", elapsed);

    Ok(())
}
