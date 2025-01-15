//! Queries a chimeric event

use anyhow::Result;
use clap::Parser;
use deepbiop::bam::chimeric::ChimericEvent;
use deepbiop::utils::interval::{GenomicInterval, Overlap};
use itertools::Itertools;
use log::debug;
use log::info;
use rayon::prelude::*;
use std::io::Write;
use std::path::{Path, PathBuf};

use deepbiop::bam::chimeric::is_chimeric_record;
use noodles::bam;
use noodles::sam;

fn main() -> Result<()> {
    println!("Hello, world!");
    Ok(())
}
