# Foul Play ![umbreon](https://play.pokemonshowdown.com/sprites/xyani/umbreon.gif)
A Pokémon battle-bot that can play battles on [Pokemon Showdown](https://pokemonshowdown.com/).

Foul Play can play single battles in all generations
though currently dynamax and z-moves are not supported.

![badge](https://github.com/pmariglia/foul-play/actions/workflows/ci.yml/badge.svg)

## Python version
Requires Python 3.11+.

## Getting Started

### Configuration

Command-line arguments are used to configure Foul Play

use `python run.py --help` to see all options.

### Running Locally

**1. Clone**

Clone the repository with `git clone https://github.com/pmariglia/foul-play.git`

**2. Install Requirements**

Install the requirements with `pip install -r requirements.txt`.

Note: Requires Rust to be installed on your machine to build the engine.

**4. Run**

Run with `python run.py`

Here is a minimal example that plays a gen9randombattle on Pokemon Showdown:
```bash
python run.py \
--websocket-uri wss://sim3.psim.us/showdown/websocket \
--ps-username 'My Username' \
--ps-password sekret \
--bot-mode search_ladder \
--pokemon-format gen9randombattle
```

### Running with Docker

**1. Clone the repository**

`git clone https://github.com/pmariglia/foul-play.git`

**2. Build the Docker image**

Use the `Makefile` to build a Docker image
```shell
make docker
```

or for a specific generation:
```shell
make docker GEN=gen4
```

**3. Run the Docker Image**
```bash
docker run --rm --network host foul-play:latest \
--websocket-uri wss://sim3.psim.us/showdown/websocket \
--ps-username 'My Username' \
--ps-password sekret \
--bot-mode search_ladder \
--pokemon-format gen9randombattle
```

## Engine

This project uses [poke-engine](https://github.com/pmariglia/poke-engine) to search through battles.
See [the engine docs](https://poke-engine.readthedocs.io/en/latest/) for more information.

The engine must be built from source if installing locally so you must have rust installed on your machine.

### Re-Installing the Engine

It is common to want to re-install the engine for different generations of Pokémon.

`pip` will used cached .whl artifacts when installing packages
and cannot detect the `--config-settings` flag that was used to build the engine.

The following command will ensure that the engine is re-installed properly:
```shell
pip uninstall -y poke-engine && pip install -v --force-reinstall --no-cache-dir poke-engine --config-settings="build-args=--features poke-engine/<GENERATION> --no-default-features"
```

Or using the Makefile:
```shell
make poke_engine GEN=<generation>
```

For example, to re-install the engine for generation 4:
```shell
make poke_engine GEN=gen4
```

## Offline Blind Ladder deployment maintenance

The canonical Blind Ladder maintenance entrypoint is:

```shell
python -m fp.data.blind_pool.maintenance --help
```

Its `build-registry`, `preflight`, `verify-artifacts`, `status`,
`resolve-consumed`, and `resolve-not-consumed` commands are offline.
Stop the canonical runtime before using them: maintenance and runtime use the
same deployment-owner guard.

- `build-registry` reads an explicit opaque membership plan and canonical
  metadata only. It does not inspect packed teams or sidecars, and it does not
  repeat team-legality validation for already provisioned artifacts.
- `preflight` observes deployment and state readiness without initializing,
  recovering, or rewriting state.
- `verify-artifacts` deliberately performs full verification of every active
  canonical artifact, one at a time.
- `status` reports the safe deployment state without changing it. An
  `accept_sent` quarantine is reported as `recovery_required` with a derived
  reconciliation case; the private challenge token is neither required nor
  displayed.

To resolve an `accept_sent` quarantine, keep the runtime stopped and investigate
the external operational evidence. If it proves the selection was consumed,
run `resolve-consumed --case <case-id> --confirm consumed`; this advances the
bag exactly as a normal room commitment would. Here, consumed means the
selection counts as having produced the accepted battle attempt or room for bag
accounting; it does not depend on battle completion, result, replay availability,
or the later battle process. If it proves no room consumed the selection, run
`resolve-not-consumed --case <case-id> --confirm not-consumed`; this releases
the same selection for retry without advancing the bag. If the evidence remains
ambiguous, leave the state unresolved. The case must match the exact current
state, and the operator's explicit disposition—not an automatic evidence check—
is authoritative.

After resolution, run `status` or `preflight` again before restarting the
runtime. These commands perform no network activity and do not display team
contents. Do not inspect or copy the private token from the state file.
