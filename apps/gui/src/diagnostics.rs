//! The CPU and memory usage of the system and of this process (under the diagnostic paths of
//! `SystemInformationDiagnosticsPlugin`), measured once per second on a thread of its own.
//!
//! On Windows, from the kernel's own counters: the CPU time of this process and of the system (`GetProcessTimes`,
//! `GetSystemTimes`) over the exact time between two measurements, the memory load of the system
//! (`GlobalMemoryStatusEx`, as in the Task Manager), and the working set of this process (`GetProcessMemoryInfo`).
//! Bevy's plugin (sysinfo) measures five times per second, and lists all processes of the system for every
//! measurement. Elsewhere, sysinfo (for this process only).

use bevy::diagnostic::{
    Diagnostic, Diagnostics, RegisterDiagnostic, SystemInformationDiagnosticsPlugin as Paths,
};
use bevy::prelude::*;
use std::sync::{Arc, Mutex};
use std::time::Duration;

const PATHS: [bevy::diagnostic::DiagnosticPath; 4] = [
    Paths::SYSTEM_CPU_USAGE,
    Paths::SYSTEM_MEM_USAGE,
    Paths::PROCESS_CPU_USAGE,
    Paths::PROCESS_MEM_USAGE,
];

pub struct UsagePlugin;

impl Plugin for UsagePlugin {
    fn build(&self, app: &mut App) {
        for path in PATHS {
            app.register_diagnostic(Diagnostic::new(path));
        }
        let latest = Latest::default();
        let measured = latest.0.clone();
        std::thread::Builder::new()
            .name("usage".into())
            .spawn(move || {
                let mut meter = Meter::new();
                loop {
                    std::thread::sleep(Duration::from_secs(1));
                    if let Some(usage) = meter.measure()
                        && let Ok(mut latest) = measured.lock()
                    {
                        *latest = Some(usage);
                    }
                }
            })
            .expect("a thread for the usage measurements");
        app.insert_resource(latest).add_systems(Update, report);
    }
}

// The usage, last measured: the CPU of the system and of this process (% of all logical processors), the memory of the
// system (% in use), and the memory of this process (GiB).
#[derive(Clone, Copy)]
struct Usage([f64; 4]);

#[derive(Resource, Default)]
struct Latest(Arc<Mutex<Option<Usage>>>);

fn report(latest: Res<'_, Latest>, mut diagnostics: Diagnostics<'_, '_>) {
    let Some(Usage(values)) = latest.0.lock().ok().and_then(|mut latest| latest.take()) else {
        return;
    };
    for (path, value) in PATHS.iter().zip(values) {
        diagnostics.add_measurement(path, || value);
    }
}

const GIB: f64 = (1u64 << 30) as f64;

#[cfg(windows)]
struct Meter {
    // The previous counters (100 ns units): system idle, kernel (with idle) and user time, the time of this process,
    // and the wall-clock time.
    previous: Option<([u64; 3], u64, std::time::Instant)>,
    processors: f64,
}

#[cfg(windows)]
impl Meter {
    fn new() -> Self {
        let processors = std::thread::available_parallelism().map_or(1, usize::from) as f64;
        Self { previous: None, processors }
    }

    fn measure(&mut self) -> Option<Usage> {
        use windows_sys::Win32::Foundation::FILETIME;
        use windows_sys::Win32::System::ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
        use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
        use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes, GetSystemTimes};

        let ticks = |t: FILETIME| (u64::from(t.dwHighDateTime) << 32) | u64::from(t.dwLowDateTime);
        let zero = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
        let (mut idle, mut kernel, mut user) = (zero, zero, zero);
        let (mut created, mut exited, mut process_kernel, mut process_user) = (zero, zero, zero, zero);
        // SAFETY: the out-pointers are valid for the calls; the pseudo handle of this process needs no closing.
        let (system_ok, process_ok) = unsafe {
            (
                GetSystemTimes(&mut idle, &mut kernel, &mut user),
                GetProcessTimes(
                    GetCurrentProcess(),
                    &mut created,
                    &mut exited,
                    &mut process_kernel,
                    &mut process_user,
                ),
            )
        };
        if system_ok == 0 || process_ok == 0 {
            return None;
        }
        let now = std::time::Instant::now();
        let system = [ticks(idle), ticks(kernel), ticks(user)];
        let process = ticks(process_kernel) + ticks(process_user);
        let previous = self.previous.replace((system, process, now));

        let mut memory = MEMORYSTATUSEX {
            dwLength: size_of::<MEMORYSTATUSEX>() as u32,
            ..unsafe { std::mem::zeroed() }
        };
        let mut counters = PROCESS_MEMORY_COUNTERS {
            cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
            ..unsafe { std::mem::zeroed() }
        };
        // SAFETY: the structures are valid and their sizes set, as the calls require.
        let (memory_ok, counters_ok) = unsafe {
            (
                GlobalMemoryStatusEx(&mut memory),
                GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb),
            )
        };

        let (system_before, process_before, before) = previous?;
        let busy = (system[1] - system_before[1]) - (system[0] - system_before[0]) + (system[2] - system_before[2]);
        let total = (system[1] - system_before[1]) + (system[2] - system_before[2]);
        let wall = (now - before).as_secs_f64() * 1e7;
        Some(Usage([
            100. * busy as f64 / (total as f64).max(1.),
            if memory_ok != 0 { f64::from(memory.dwMemoryLoad) } else { 0. },
            100. * (process - process_before) as f64 / (wall * self.processors).max(1.),
            if counters_ok != 0 { counters.WorkingSetSize as f64 / GIB } else { 0. },
        ]))
    }
}

#[cfg(not(windows))]
struct Meter {
    system: sysinfo::System,
    pid: Option<sysinfo::Pid>,
}

#[cfg(not(windows))]
impl Meter {
    fn new() -> Self {
        Self { system: sysinfo::System::new(), pid: sysinfo::get_current_pid().ok() }
    }

    fn measure(&mut self) -> Option<Usage> {
        use sysinfo::{CpuRefreshKind, MemoryRefreshKind, ProcessRefreshKind, ProcessesToUpdate};
        let pid = self.pid?;
        let system = &mut self.system;
        system.refresh_cpu_specifics(CpuRefreshKind::nothing().with_cpu_usage());
        system.refresh_memory_specifics(MemoryRefreshKind::nothing().with_ram());
        system.refresh_processes_specifics(
            ProcessesToUpdate::Some(&[pid]),
            true,
            ProcessRefreshKind::nothing().with_cpu().with_memory(),
        );
        let process = system.process(pid)?;
        Some(Usage([
            f64::from(system.global_cpu_usage()),
            100. * system.used_memory() as f64 / (system.total_memory() as f64).max(1.),
            f64::from(process.cpu_usage()) / system.cpus().len().max(1) as f64,
            process.memory() as f64 / GIB,
        ]))
    }
}
