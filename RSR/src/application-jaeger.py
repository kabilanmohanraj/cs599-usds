from lib.config import ClusterConfig, BoundBox, TimeWindow
from lib.utils import GoogleArchive, Data
from lib.models import MockModel
import ray
from collections import defaultdict
import time as clock
import argparse
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from lib.trace_utils import parse_dict_ctx, get_parent_context


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sample application for running models")
    parser.add_argument("--urls", dest="urlCount", help="Number of urls to download", type=int, required=True)
    args = parser.parse_args()
    trace.set_tracer_provider(
        TracerProvider(
            resource=Resource.create({SERVICE_NAME: "RSR"})
        )
    )
    jaeger_exporter = JaegerExporter(
        agent_host_name="localhost",
        agent_port=6831,
    )
    trace.get_tracer_provider().add_span_processor(
        BatchSpanProcessor(jaeger_exporter)
    )
    runtime_env = {"working_dir": "."}
    bound_box = BoundBox(12, 31, 12, 31)
    time_window = TimeWindow(1)

    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span("main-start") as main:
        ray.init(address="auto", runtime_env=runtime_env)
        global_timers = defaultdict(list)
        global_timers['e2e'].append(clock.time())
        global_timers['clusterCreation'].append(clock.time())
        c = ClusterConfig(len(ray.nodes()), ray.cluster_resources()['CPU'], int(ray.cluster_resources()['CPU']))

        context = main.get_span_context()
        context_dict = {'traceId': context.trace_id,
                 'spanId': context.span_id}

        # with tracer.start_as_current_span("initialize_workers"):
        c.initialize_workers(context_dict)

        global_timers['clusterCreation'].append(clock.time())
        global_timers['archiveDownload'].append(clock.time())

        with tracer.start_as_current_span("Google_Archive") as garch:
            archive = GoogleArchive(c, context_dict)

            context = garch.get_span_context()
            context_dict = {'traceId': context.trace_id,
                 'spanId': context.span_id}

            archive.context_dict = context_dict

            f = archive.query(bound_box, time_window).fetch_urls(args.urlCount)
            f.download()
            imageData = archive.redistribute() # (TODO) yet to be done


        global_timers['archiveDownload'].append(clock.time())
        global_timers['redistribute'].append(clock.time())

        # with tracer.start_as_current_span("redistribute"):

        global_timers['redistribute'].append(clock.time())
        global_timers['applyModel'].append(clock.time())

        # with tracer.start_as_current_span("model_application") as ma:
        r_model = MockModel()
        context = main.get_span_context()
        context_dict = {'traceId': context.trace_id,
                'spanId': context.span_id}
        imageData.apply_model(r_model, ctx_dic=context_dict)

        global_timers['applyModel'].append(clock.time())
        global_timers['collectFiles'].append(clock.time())

        # with tracer.start_as_current_span("collect_data"):
        imageData.collect_data(context_dict=context_dict)

        global_timers['collectFiles'].append(clock.time())
        global_timers['e2e'].append(clock.time())

        for key in global_timers.keys():
            print(f'{key}: {global_timers[key][1] - global_timers[key][0]}')