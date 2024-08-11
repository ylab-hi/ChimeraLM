use ahash::HashMap;
use anyhow::Result;
use clap::Parser;
use deepbiop::bam::chimeric::ChimericEvent;
use deepbiop::utils::interval::Overlap;
use itertools::Itertools;
use log::info;
use rayon::prelude::*;
use std::io::Write;
use std::path::PathBuf;

use noodles::bam;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    /// path to the clean bam
    #[arg(long="cbam", value_name = "clean bam", action = clap::ArgAction::Append)]
    cbam: Vec<PathBuf>,

    /// path to dirty bam
    #[arg(long="dbam", value_name = "dirty bam", action = clap::ArgAction::Append)]
    dbam: Vec<PathBuf>,

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

fn is_same_chimeric_event(
    cbam_chimeric_event: &ChimericEvent,
    dbam_chimeric_event: &ChimericEvent,
) -> bool {
    cbam_chimeric_event.len() == dbam_chimeric_event.len()
        && cbam_chimeric_event.name == dbam_chimeric_event.name
        && cbam_chimeric_event
            .intervals
            .iter()
            .zip_eq(&dbam_chimeric_event.intervals)
            .all(|(event1, event2)| event1.overlap(event2))
}

fn check_chimeric_events_sup(
    cbam_chimeric_event: &ChimericEvent,
    dbam_chimeric_events: &[ChimericEvent],
) -> bool {
    // check dbam chimeric events include cbam, and early stop
    dbam_chimeric_events
        .par_iter()
        .any(|dbam_chimeric_event| is_same_chimeric_event(cbam_chimeric_event, dbam_chimeric_event))
}

fn write_results(results: &HashMap<PathBuf, HashMap<String, Vec<String>>>) -> Result<()> {
    for (cbam_path, read_sups) in results.iter() {
        let output_path = cbam_path.with_extension("sup.txt");
        info!("writing to {:?}", output_path);

        let mut buf_writer = std::io::BufWriter::new(std::fs::File::create(output_path)?);

        for (read_name, sup_paths) in read_sups.iter() {
            let line = format!(
                "{}\t{}\t{}\n",
                read_name,
                sup_paths.len(),
                sup_paths.iter().join(",")
            );
            buf_writer.write_all(line.as_bytes())?;
        }
    }
    Ok(())
}

fn annote(cbam: &[PathBuf], dbam: &[PathBuf], threads: Option<usize>) -> Result<()> {
    let cbam_chimeric_events_per_bam: HashMap<PathBuf, Vec<ChimericEvent>> = cbam
        .par_iter()
        .map(|path| {
            let mut chimeric_events: Vec<ChimericEvent> =
                deepbiop::bam::chimeric::create_chimeric_events_from_bam(
                    path,
                    threads,
                    None::<fn(&bam::Record) -> bool>,
                )
                .unwrap();

            chimeric_events.par_iter_mut().for_each(|event| {
                event
                    .intervals
                    .sort_by_key(|interval| interval.chr.to_string());
            });

            (path.clone(), chimeric_events)
        })
        .collect();

    let dbam_chimeric_events_per_bam: HashMap<PathBuf, Vec<ChimericEvent>> = dbam
        .par_iter()
        .map(|path| {
            let mut chimeric_events = deepbiop::bam::chimeric::create_chimeric_events_from_bam(
                path,
                threads,
                None::<fn(&bam::Record) -> bool>,
            )
            .unwrap();

            chimeric_events.par_iter_mut().for_each(|event| {
                event
                    .intervals
                    .sort_by_key(|interval| interval.chr.to_string());
            });

            (path.clone(), chimeric_events)
        })
        .collect();

    let all_sups_result: HashMap<PathBuf, HashMap<String, Vec<String>>> =
        cbam_chimeric_events_per_bam
            .par_iter()
            .map(|(path, events)| {
                let read_sups = events
                    .par_iter()
                    .map(|event| {
                        let sup_paths = dbam_chimeric_events_per_bam
                            .par_iter()
                            .filter_map(|(dbam_path, dbam_chimeric_events)| {
                                if check_chimeric_events_sup(event, dbam_chimeric_events) {
                                    Some(
                                        dbam_path
                                            .file_name()
                                            .unwrap()
                                            .to_string_lossy()
                                            .to_string(),
                                    )
                                } else {
                                    None
                                }
                            })
                            .collect();

                        (event.name.as_ref().unwrap().to_string().clone(), sup_paths)
                    })
                    .collect::<HashMap<String, Vec<String>>>();

                (path.clone(), read_sups)
            })
            .collect();
    write_results(&all_sups_result)?;
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

    annote(&cli.cbam, &cli.dbam, None).unwrap();

    let elapsed = start.elapsed();
    log::info!("elapsed time: {:.2?}", elapsed);
    Ok(())
}
