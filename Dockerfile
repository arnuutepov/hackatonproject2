# Official check image: Python 3.11, no network needed at run time.
# The base image is pinned by digest: the one our Docker runs of 22.11 resolved python:3.11-slim to
# (artifacts/logs/docker_check.log). To move to a newer base, pull the tag and take its digest:
#   docker inspect --format '{{index .RepoDigests 0}}' python:3.11-slim
FROM python:3.11-slim@sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# Test-only dependencies, fully pinned. The environment itself uses the standard library only.
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

COPY . .
# Readable by any user, writable by none but root: the image runs as an unprivileged user.
RUN chmod -R a+rX,go-w /app

# The jury runs the container as the image USER, without root (organizers' answer of 21.11).
# /app is then read-only for the check, which writes runs/, artifacts/ and regenerated episodes
# under $TMPDIR/auditenv-out (or AUDITENV_OUT) and compares the episodes with the committed ones.
# Also passes with --read-only --tmpfs /tmp (make docker-check-hardened).
ENV HOME=/tmp
USER 10001:10001

# No `make` in slim images (audit F06): the entrypoint is plain Python.
CMD ["python", "-m", "auditenv.run", "check", "--with-tests"]
