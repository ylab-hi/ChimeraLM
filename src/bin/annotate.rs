use ahash::HashMap;
use anyhow::Result;
use clap::Parser;
use deepbiop::bam::chimeric::ChimericEvent;
use deepbiop::utils::interval::{GenomicInterval, Overlap};
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

const OVERLAP_THRESHOLD: usize = 5000;

fn check_overlap(
    interval1: &GenomicInterval,
    interval2: &GenomicInterval,
    threshold: usize,
) -> bool {
    // Early return if chromosomes don't match
    if interval1.chr != interval2.chr {
        return false;
    }

    // Check direct overlap first
    if interval1.overlap(interval2) {
        return true;
    }

    // Calculate distance between intervals
    if interval1.end < interval2.start {
        interval2.start - interval1.end < threshold
    } else {
        interval1.start - interval2.end < threshold
    }
}

fn is_same_chimeric_event(
    bam_chimeric_event1: &ChimericEvent,
    bam_chimeric_event2: &ChimericEvent,
) -> bool {
    bam_chimeric_event1.len() == bam_chimeric_event2.len()
        && bam_chimeric_event1
            .intervals
            .iter()
            .zip_eq(&bam_chimeric_event2.intervals)
            .all(|(interval1, interval2)| check_overlap(interval1, interval2, OVERLAP_THRESHOLD))
}

fn check_chimeric_events_sup(
    dbam_chimeric_event: &ChimericEvent,
    cbam_chimeric_events: &[ChimericEvent],
) -> bool {
    // check dbam chimeric events include cbam, and early stop
    cbam_chimeric_events
        .par_iter()
        .any(|cbam_chimeric_event| is_same_chimeric_event(dbam_chimeric_event, cbam_chimeric_event))
}

fn write_results(results: &HashMap<PathBuf, HashMap<String, Vec<String>>>) -> Result<()> {
    for (cbam_path, read_sups) in results.iter() {
        let output_path = cbam_path.with_extension("sup.txt");
        info!("writing {} reads to {:?}", read_sups.len(), output_path);

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
                event.intervals.sort_by_key(|interval| {
                    (interval.chr.to_string(), interval.start, interval.end)
                });
            });

            (path.clone(), chimeric_events)
        })
        .collect();

    for (path, events) in cbam_chimeric_events_per_bam.iter() {
        info!("{:?} collect {} chimeric events", path, events.len());
    }

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

    for (path, events) in dbam_chimeric_events_per_bam.iter() {
        info!("{:?} collect {} chimeric events", path, events.len());
    }

    let all_sups_result: HashMap<PathBuf, HashMap<String, Vec<String>>> =
        dbam_chimeric_events_per_bam
            .par_iter()
            .map(|(path, events)| {
                let read_sups = events
                    .par_iter()
                    .map(|event| {
                        let sup_paths = cbam_chimeric_events_per_bam
                            .par_iter()
                            .filter_map(|(cbam_path, cbam_chimeric_events)| {
                                if check_chimeric_events_sup(event, cbam_chimeric_events) {
                                    log::debug!(
                                        "found sup for {} in {:?}",
                                        event.name.as_ref().unwrap().to_string(),
                                        cbam_path
                                    );
                                    Some(
                                        cbam_path
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_overlap_direct_overlap() {
        let interval1 = GenomicInterval {
            chr: "chr1".into(),
            start: 100,
            end: 200,
        };
        let interval2 = GenomicInterval {
            chr: "chr1".into(),
            start: 150,
            end: 250,
        };
        let threshold = 0;
        assert!(check_overlap(&interval1, &interval2, threshold));
    }

    #[test]
    fn test_check_overlap_no_overlap_within_threshold() {
        let interval1 = GenomicInterval {
            chr: "chr1".into(),
            start: 100,
            end: 200,
        };
        let interval2 = GenomicInterval {
            chr: "chr1".into(),
            start: 300,
            end: 400,
        };
        let threshold = 100;

        println!("overlap: {}", interval1.overlap(&interval2));

        assert!(!check_overlap(&interval1, &interval2, threshold));
    }
}
