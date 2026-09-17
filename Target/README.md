# v4 Emulation Target

A small, self-contained Windows-domain network you can attack over HTTP. It is
the live counterpart to the training simulator: both run the same world model
(`env/kill_chain.py`) and the same rule-based detection (`env/detection.py`), so
the target behaves identically to the environment the agent trained in.

The network is three hosts in one Active Directory domain:

| Host | Role | Services |
|---|---|---|
| `user01` | workstation | mail, web app, workstation login |
| `srv01` | file server | SMB / shares, service control |
| `dc01` | domain controller | Kerberos, LDAP, admin / replication |

Everything the target returns — credentials, hashes, tickets, records — is a
**synthetic fixture for the range**. There are no real secrets here.

## Running it

```bash
# development
python Target/mock_server.py                 # serves on :5000

# production WSGI (if waitress is installed)
waitress-serve --port=5000 --call Target.mock_server:create_app
```

Or with Docker:

```bash
docker build -t v4-target Target/
docker run -p 5000:5000 v4-target
```

## Design in one line

State transitions go through the shared model (one source of truth, so the
simulator and the server never diverge); the service layer (`Target/services.py`)
only decides what a real service *returns* when a technique succeeds.

## API

### Session & state
| Method | Path | Purpose |
|---|---|---|
| `POST` | `/session` | create a session; returns a bearer token |
| `POST` | `/reset` | start a fresh episode for the session |
| `GET`  | `/state` | full authoritative network state |
| `GET`  | `/health` | liveness + inventory |

Every call below carries `Authorization: Bearer <session-token>`.

### Attacker interface
| Method | Path | Purpose |
|---|---|---|
| `POST` | `/attempt` | run a technique by name (the RL interface) |
| `POST` | `/svc/<host>/<service>/<action>` | drive a specific service like a real one |
| `GET`  | `/network/hosts` | inventory visible to the attacker (gated on discovery) |

A successful action returns an `artifact`: the realistic result a real service
would give — a dumped credential table, a `$krb5tgs$` ticket, an NTDS dump, and
so on.

### Defender interface
| Method | Path | Purpose |
|---|---|---|
| `GET` | `/siem/events` | the ATT&CK-mapped security events, with SOC suspicion |
| `GET` | `/siem/export` | the same log in an ECS-like schema a SIEM can ingest |

## Example: a manual intrusion

```bash
BASE=http://localhost:5000
TOK=$(curl -s -XPOST $BASE/session | jq -r .session)
AUTH="Authorization: Bearer $TOK"
curl -s -XPOST $BASE/reset -H "$AUTH" -d '{"sqli_available":true}' >/dev/null

# dump the web-app credential table via SQL injection
curl -s -XPOST $BASE/svc/user01/webapp/query -H "$AUTH" | jq .artifact

# ... escalate, then read the SOC's view of what you did
curl -s $BASE/siem/events -H "$AUTH" | jq '{suspicion, fired_rules}'
```
