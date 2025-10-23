#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example script for running SuperCon workchains using the SubmissionController.

This script demonstrates how to:
1. Set up a SuperCon controller
2. Submit workchains in batches
3. Monitor progress
4. Handle different input types
"""

import time
from aiida import load_profile, orm
from aiida_epw_workflows.workflows.controllers.supercon_complete import (
    SuperConWorkChainController,
    create_supercon_controller
)


def setup_example_groups():
    """Set up example groups for demonstration."""
    # Create parent group for input structures/workchains
    parent_group, created = orm.Group.collection.get_or_create("supercon_inputs")

    # Create group for SuperCon workchains
    workchain_group, _ = orm.Group.collection.get_or_create("supercon_workchains")

    print(f"Parent group: {parent_group.label}")
    print(f"Workchain group: {workchain_group.label}")

    return parent_group, workchain_group


def create_controller():
    """Create a SuperCon controller with your specific configuration."""

    # Define your codes (replace with your actual code labels)
    codes = {
        "pw_code": "pw@lumi",           # Quantum ESPRESSO pw.x
        "ph_code": "ph@lumi",           # Quantum ESPRESSO ph.x
        "q2r_code": "q2r@lumi",         # Quantum ESPRESSO q2r.x
        "matdyn_code": "matdyn@lumi",   # Quantum ESPRESSO matdyn.x
    }

    # Create controller using convenience function
    controller = create_supercon_controller(
        parent_group_label="supercon_inputs",
        group_label="supercon_workchains",
        max_concurrent=3,  # Run max 3 workchains simultaneously
        protocol="moderate",
        **codes,
        # Optional: customize resource settings
        max_wallclock_seconds=7200,  # 2 hours
        account="project_465000106",
        queue_name="debug",
        num_machines=1,
        num_mpiprocs_per_machine=16,
        num_cores_per_machine=16,
    )

    return controller


def run_controller_basic():
    """Run the controller with basic monitoring."""
    print("=== Basic Controller Run ===")

    controller = create_controller()

    # Check initial status
    summary = controller.get_status_summary()
    print(f"Initial status: {summary}")

    if summary['still_to_run'] == 0:
        print("No workchains to submit!")
        return

    # Submit workchains with monitoring
    controller.submit_with_monitoring(verbose=True, sleep_interval=60)


def run_controller_manual():
    """Run the controller with manual control."""
    print("=== Manual Controller Run ===")

    controller = create_controller()

    # Manual submission loop
    while True:
        summary = controller.get_status_summary()
        print(f"\nStatus: {summary['active_slots']}/{summary['max_concurrent']} active, "
              f"{summary['available_slots']} available, {summary['still_to_run']} remaining")

        if summary['still_to_run'] == 0:
            print("All workchains submitted!")
            break

        if summary['available_slots'] > 0:
            # Submit new batch
            submitted = controller.submit_new_batch(verbose=True)
            if submitted:
                print(f"Submitted {len(submitted)} new workchains:")
                for extras, node in submitted.items():
                    print(f"  {extras} -> PK: {node.pk}")
        else:
            print("No available slots, waiting...")

        # Wait before next iteration
        time.sleep(30)


def run_controller_dry_run():
    """Run a dry run to see what would be submitted."""
    print("=== Dry Run ===")

    controller = create_controller()

    # Do a dry run to see what would be submitted
    would_submit = controller.submit_new_batch(dry_run=True, verbose=True)

    print(f"\nWould submit {len(would_submit)} workchains:")
    for extras in would_submit.keys():
        print(f"  {extras}")


def check_results():
    """Check the results of completed workchains."""
    print("=== Checking Results ===")

    controller = create_controller()

    # Get all submitted workchains
    all_submitted = controller.get_all_submitted_processes()

    print(f"Total submitted workchains: {len(all_submitted)}")

    # Check status of each workchain
    for extras, workchain in all_submitted.items():
        status = workchain.process_state.value
        print(f"{extras}: {status}")

        # If completed, show some outputs
        if status == "finished":
            try:
                # Check for specific outputs (adjust based on your workchain outputs)
                if hasattr(workchain.outputs, 'parameters'):
                    params = workchain.outputs.parameters.get_dict()
                    print(f"  Parameters: {params}")
            except Exception as e:
                print(f"  Error getting outputs: {e}")


def main():
    """Main execution function."""
    # Load AiiDA profile
    load_profile()

    # Set up groups
    setup_example_groups()

    # Choose which mode to run
    mode = "basic"  # Options: "basic", "manual", "dry_run", "check_results"

    if mode == "basic":
        run_controller_basic()
    elif mode == "manual":
        run_controller_manual()
    elif mode == "dry_run":
        run_controller_dry_run()
    elif mode == "check_results":
        check_results()
    else:
        print("Invalid mode. Choose from: basic, manual, dry_run, check_results")


if __name__ == "__main__":
    main()