//! The latest console message, shared with tracing's worker threads.

use bevy::log::{BoxedLayer, Level, tracing, tracing_subscriber::Layer};
use bevy::prelude::*;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct LogLine {
    pub text: String,
    pub level: Level,
}

#[derive(Resource, Clone, Default)]
pub struct Console(pub Arc<Mutex<Option<LogLine>>>);

pub fn capture(app: &mut App) -> Option<BoxedLayer> {
    let console = Console::default();
    app.insert_resource(console.clone());
    Some(console.boxed())
}

impl<S: tracing::Subscriber> Layer<S> for Console {
    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        _: bevy::log::tracing_subscriber::layer::Context<'_, S>,
    ) {
        let mut visitor = MessageVisitor(None);
        event.record(&mut visitor);
        if let Some(text) = visitor.0
            && let Ok(mut latest) = self.0.lock()
        {
            *latest = Some(LogLine { text, level: *event.metadata().level() });
        }
    }
}

struct MessageVisitor(Option<String>);

impl tracing::field::Visit for MessageVisitor {
    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        if field.name() == "message" {
            self.0 = Some(value.to_owned());
        }
    }

    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            self.0 = Some(format!("{value:?}"));
        }
    }
}
