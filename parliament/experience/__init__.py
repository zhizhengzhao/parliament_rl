"""Science Parliament — experience accumulation (reserved).

After a parliament session completes, an experience agent can review the
discussion and extract structured lessons into .md files.  These lessons
can be loaded into future sessions to improve scientist performance.

Enable via config.EXPERIENCE_ENABLED = True.

Planned interface:
    from experience import generate_experience
    await generate_experience(session_data, model, output_dir)
"""
