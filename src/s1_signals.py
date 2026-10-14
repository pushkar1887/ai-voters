"""Temporary entry point for runs started under the old name. Delete when no run uses it."""
import runpy

runpy.run_module("src.stages.stage1_private_signals", run_name="__main__")
