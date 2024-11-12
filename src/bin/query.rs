//! Queries a chimeric event
//!
//! The input BAM must have an index in the same directory.
//!
//! The result matches the output of `samtools view <src> <region>`.

use ahash::HashMap;
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
