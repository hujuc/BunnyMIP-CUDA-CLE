# Accessing `banana.ua.pt` — UA VPN + SSH (guide for new users)

All development runs on the **`banana.ua.pt`** server (CUDA/GPUs). It is only
reachable from **inside the UA network**, so you must connect to the **UA VPN**
(Check Point) first. `banana.ua.pt` does not even resolve via DNS without the VPN.

> **Day-to-day summary** (once everything is installed):
> 1. Connect the VPN in the browser (portal `https://go.ua.pt`).
> 2. `sudo bash scripts/vpn-up.sh`  ← applies the *fix* (see §5).
> 3. `ssh banana`  → you're on the server.

Tested on **Ubuntu 24.04** (host). The server is **Ubuntu 18.04, CUDA 10.2**,
GPUs GeForce GT 710 + GTX 1660. Group: **cle08** (`banana.ua.pt = 192.168.8.210`).

---

## 1. Prerequisites

- A UA Universal User account (`<user>@ua.pt`) with teleworking access.
- The group's server credentials (`cleNN` + password, requested from the instructor).
- `sudo` on your laptop.
- ⚠️ **Forbidden** to use the *VSCode Remote-SSH* extension on banana (shared server;
  assignment rule). Develop **locally** and sync with `rsync`/`git`.

---

## 2. Install the VPN client (one time) — ChrootVPN

UA uses Check Point. The standalone CLI client (`snx`) is legacy and **gets rejected
by the gateway** (it fails with `Connection aborted`). The approach that works is
**ChrootVPN** by Rui Ribeiro, which builds the Check Point client (CShell + SNX) in a
Debian *chroot* and exposes a local agent on `localhost:14186`. It is the method
referenced in UA's own documentation.

```bash
# 1) enable the i386 architecture (Check Point requirement)
sudo dpkg --add-architecture i386

# 2) get the script
curl -fsSL https://raw.githubusercontent.com/ruyrybeyro/chrootvpn/master/vpn.sh -o /tmp/cpvpn.sh

# 3) install (builds the Debian chroot + CShell; takes a few minutes)
sudo bash /tmp/cpvpn.sh -i --vpn=go.ua.pt
```

After installation:

1. **Accept the local certificate** of the CShell agent: open `https://localhost:14186/id`
   in the browser and accept the certificate (not needed in Firefox).
2. **Start the VPN**: open `https://go.ua.pt`, log in with `<user>@ua.pt` + password.
   The portal launches CShell, which creates the **`tunsnx`** interface and assigns an
   *Office Mode IP* (e.g. `172.18.6.x`). You should see "**You are connected**".

> ChrootVPN registers a service; next times you only need the browser step
> (`https://go.ua.pt`) to connect.

---

## 3. Set up SSH access to banana (one time)

With the VPN connected **and** `scripts/vpn-up.sh` applied (§5):

```bash
# generate a dedicated key
ssh-keygen -t ed25519 -f ~/.ssh/banana_cle08 -C "cle08@banana"

# install the key on the server (asks for the group password ONCE)
ssh-copy-id -i ~/.ssh/banana_cle08.pub -o IdentitiesOnly=yes cle08@banana.ua.pt
```

Convenient shortcut in `~/.ssh/config`:

```
Host banana
    HostName banana.ua.pt
    User cle08
    IdentityFile ~/.ssh/banana_cle08
    IdentitiesOnly yes
```

From now on: **`ssh banana`** logs in without a password.

---

## 4. Connecting (every session)

```bash
# 1) browser: https://go.ua.pt  -> log in -> "connected"
# 2) apply the network fix (needs sudo):
sudo bash scripts/vpn-up.sh
# 3) work:
ssh banana
```

`scripts/vpn-up.sh` verifies at the end: `banana.ua.pt -> <ip>`, `banana:22 OPEN`,
`general internet: OK`.

---

## 5. Why `scripts/vpn-up.sh`? (what breaks and how it is fixed)

When you connect, Check Point goes into **full tunnel** and, in this setup, **nothing
works** until the *fix* is applied. Three problems, all handled by the script:

1. **General internet dies (full tunnel).** The gateway pushes routes covering the
   entire IPv4 space through `tunsnx`, but UA does not provide internet egress.
   → *Fix:* **split tunnel** — only UA networks (`193.136/16`, `193.137/16` except the
   gateway IP, `192.168.8.0/24` for banana, DNS, `10/8`) go through the tunnel;
   everything else uses the normal physical network.

2. **`*.ua.pt` DNS does not resolve.** Check Point puts the UA internal resolvers in
   `resolv.conf`, but `systemd-resolved` ignores them.
   → *Fix:* `resolvectl dns tunsnx 192.168.227.20 192.168.227.21` +
   `resolvectl domain tunsnx '~ua.pt'`.

3. **(machine-specific) Docker collision.** The *Office Mode IP* lands in `172.18.x`,
   which **collides with a Docker bridge on `172.18.0.0/16`**. Docker's `MASQUERADE`
   rule catches the tunnel packets and rewrites their *source* to the LAN IP → the
   gateway drops the replies (`ping`/`ssh` fail even though the tunnel is "up").
   Symptom: in `tcpdump -i tunsnx` packets leave with the LAN IP instead of the Office
   Mode IP.
   → *Fix:* `iptables -t nat -I POSTROUTING 1 -o tunsnx -j ACCEPT` (exempts **all**
   tunnel traffic from any host NAT). This rule is harmless even without Docker, so
   the script always applies it.

The script also restores `rp_filter=0` and ensures the gateway IP (`go.ua.pt`) leaves
through the physical network (otherwise it would loop). The routes are reset by Check
Point on every connection, which is why you must run the script **every time you
connect**.

> If you recreate your Docker network, the bridge name changes; update the `DOCKER_BR`
> variable at the top of `scripts/vpn-up.sh` (or ignore it — the `ACCEPT` rule already
> covers the essential part).

---

## 6. Troubleshooting

| Symptom | Likely cause / action |
|---|---|
| `banana.ua.pt` doesn't resolve | VPN down, or DNS not applied → run `scripts/vpn-up.sh` |
| Connects but I lose internet | You didn't run `vpn-up.sh` (full tunnel) |
| `ssh banana` asks for a password | Key not installed → repeat the `ssh-copy-id` (§3) |
| `snx: Connection aborted` | You're using the standalone `snx` — use ChrootVPN (§2) |
| `tunsnx` never appears | The `https://go.ua.pt` portal login didn't complete |
| Packets leave with the wrong IP in `tcpdump -i tunsnx` | NAT collision → the `iptables ... -o tunsnx -j ACCEPT` rule in `vpn-up.sh` fixes it |

---

## 7. Disconnecting

Disconnect the VPN via the `https://go.ua.pt` portal (or via the CShell agent). The
routes/NAT added by `vpn-up.sh` disappear together with the `tunsnx` interface.
