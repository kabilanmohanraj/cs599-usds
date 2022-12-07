from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags
from opentelemetry import trace

def get_parent_context(trace_id, span_id):
    parent_context = SpanContext(
        trace_id=trace_id,
        span_id=span_id,
        is_remote=True,
        trace_flags=TraceFlags(0x01)
    )
    return trace.set_span_in_context(NonRecordingSpan(parent_context))