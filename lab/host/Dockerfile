# A deliberately weak Linux host. Debian slim plus the services a real host runs.
FROM debian:12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        openssh-server sudo cron rsyslog procps iproute2 net-tools \
        python3 python3-minimal ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# sshd needs these to start; password auth is enabled on purpose
RUN mkdir -p /run/sshd /var/log \
    && sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin no/' /etc/ssh/sshd_config \
    && sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication yes/' /etc/ssh/sshd_config \
    && sed -i 's/^#\?MaxAuthTries.*/MaxAuthTries 30/' /etc/ssh/sshd_config

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 22
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
