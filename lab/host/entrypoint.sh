#!/bin/bash
# Build the host's weaknesses from the compose environment, then run sshd.
set -e

# --- accounts, from LAB_USERS="name:password,name:password" -------------------
IFS=',' read -ra PAIRS <<< "${LAB_USERS:-}"
for p in "${PAIRS[@]}"; do
    u="${p%%:*}"; pw="${p#*:}"
    [ -z "$u" ] && continue
    useradd -m -s /bin/bash "$u" 2>/dev/null || true
    echo "$u:$pw" | chpasswd
done

# --- a sudo rule that permits a shell escape (T1068) --------------------------
# find -exec is the classic: it runs arbitrary commands as root.
if [ "${LAB_SUDO_MISCONFIG:-0}" = "1" ]; then
    echo 'alice ALL=(ALL) NOPASSWD: /usr/bin/find' > /etc/sudoers.d/lab
    chmod 440 /etc/sudoers.d/lab
fi

# --- documents worth collecting, and a credential left in a config file -------
mkdir -p /srv/share /etc/app
for i in 0 1 2 3 4 5; do
    printf 'quarterly report %s\n' "$i" > "/srv/share/report_${i}.txt"
done
cat > /etc/app/config.ini <<'CFG'
[database]
host = db01.internal
user = svc_backup
password = backup
CFG
chmod 644 /etc/app/config.ini

# --- a real log to clear (T1070) ---------------------------------------------
for i in $(seq 1 40); do echo "$(date -Is) lab: audit event $i" >> /var/log/auth.log; done

# --- a cron directory that can be written for persistence (T1053.003) --------
mkdir -p /etc/cron.d && chmod 777 /etc/cron.d

service rsyslog start >/dev/null 2>&1 || true
service cron start    >/dev/null 2>&1 || true
exec /usr/sbin/sshd -D -e
