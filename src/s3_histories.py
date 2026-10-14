"""Temporary entry point for runs started under the old name. Delete when no run uses it."""
import runpy

runpy.run_module("src.stages.stage3_histories", run_name="__main__")
