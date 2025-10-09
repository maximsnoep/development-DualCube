pub mod formats {
    pub mod backwards_compatability;
    pub mod dcube;
    pub mod dotgraph;
    pub mod flag;
    pub mod nlr;
    pub mod obj;
}

pub trait Export {
    fn export(solution: &dualcube::prelude::Solution, path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>>;
}

pub trait Import {
    fn import(path: &std::path::Path) -> Result<dualcube::prelude::Solution, Box<dyn std::error::Error>>;
}

pub use crate::formats::{backwards_compatability::BackwardsCompatibility, dcube::Dcube, dotgraph::Dotgraph, flag::Flag, nlr::Nlr, obj::Obj};
use dualcube::prelude::Solution;
use std::{path::PathBuf, sync::Arc};

pub fn import_solution(path: PathBuf) -> Solution {
    match path.extension().unwrap().to_str() {
        Some("obj") => {
            let mesh = match mehsh::mesh::connectivity::Mesh::from_obj(&path) {
                Ok(res) => Arc::new(res.0),
                Err(err) => {
                    panic!("Error while parsing OBJ file {path:?}: {err:?}");
                }
            };
            Solution::new(mesh.clone())
        }
        Some("stl") => {
            let mesh = match mehsh::mesh::connectivity::Mesh::from_stl(&path) {
                Ok(res) => Arc::new(res.0),
                Err(err) => {
                    panic!("Error while parsing STL file {path:?}: {err:?}");
                }
            };
            Solution::new(mesh.clone())
        }
        Some("dcube") => {
            if let Ok(sol) = Dcube::import(&path) {
                sol
            } else {
                panic!("Error while parsing DCube file {path:?}");
            }
        }
        Some("save") => {
            println!("Loading save file {path:?}");
            if let Ok(sol) = BackwardsCompatibility::import(&path) {
                sol
            } else {
                panic!("Error while parsing save file {path:?}");
            }
        }
        _ => {
            panic!("Unsupported file extension for {path:?}");
        }
    }
}
