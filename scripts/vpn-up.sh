#!/bin/bash
# =============================================================================
# vpn-up.sh - Corrigir a VPN da UA (ChrootVPN/SNX) apos ligar pelo portal.
#
# Contexto: o cliente Check Point liga em "Network mode" (full tunnel) e:
#   - encaminha TODO o trafego pelo tunel -> a UA nao da saida p/ internet;
#   - configura DNS interno mas o systemd-resolved ignora-o;
#   - o Office Mode IP (172.18.x) COLIDE com a bridge Docker 172.18.0.0/16,
#     cujo MASQUERADE reescreve o source do tunel p/ o IP da LAN -> gateway
#     descarta as respostas.
#
# Este script: isenta o tunel do NAT do Docker, faz split tunnel (so redes UA
# pelo tunel), corrige rp_filter e o DNS de *.ua.pt. Correr a CADA ligacao:
#
#     1) ligar a VPN no browser (portal go.ua.pt) ate "connected"
#     2) sudo bash scripts/vpn-up.sh
# =============================================================================
set -u
TUN=tunsnx
DEV_PHYS=$(ip route show default | awk '/default/{print $5; exit}')
GW_PHYS=$(ip route show default | awk '/default/{print $3; exit}')
VPN_GW=193.137.173.235          # go.ua.pt (sai pela rede fisica)
DOCKER_BR=br-80698eda3332       # bridge Docker que colide com 172.18/16

ip link show "$TUN" &>/dev/null || { echo "ERRO: VPN nao ligada ($TUN ausente). Liga no browser primeiro."; exit 1; }
IP=$(ip -4 addr show "$TUN" | awk '/inet /{print $2; exit}' | cut -d/ -f1)
echo "==> office-mode IP=$IP  | fisica: $DEV_PHYS via $GW_PHYS"

echo "==> [1] isentar o tunel de TODO o NAT (corrige a colisao Docker 172.18/16)"
iptables -t nat -D POSTROUTING -o "$TUN" -j ACCEPT 2>/dev/null
iptables -t nat -I POSTROUTING 1 -o "$TUN" -j ACCEPT

echo "==> [2] rp_filter=0"
for i in all default "$TUN" "$DEV_PHYS"; do sysctl -qw "net.ipv4.conf.$i.rp_filter=0" 2>/dev/null; done

echo "==> [3] split tunnel: gateway pela fisica; so redes UA pelo tunel"
ip route replace "$VPN_GW/32" via "$GW_PHYS" dev "$DEV_PHYS"
ip route flush table main dev "$TUN"
ip route del 172.18.0.0/16 dev "$DOCKER_BR" 2>/dev/null
ip route replace 172.18.6.144/32 dev "$TUN" src "$IP"
for net in \
  193.136.0.0/16 \
  193.137.0.0/17 193.137.128.0/19 193.137.160.0/21 193.137.168.0/22 \
  193.137.172.0/24 193.137.173.0/25 193.137.173.128/26 193.137.173.192/27 \
  193.137.173.224/29 193.137.173.232/31 193.137.173.234/32 193.137.173.236/30 \
  193.137.173.240/28 193.137.174.0/23 193.137.176.0/20 193.137.192.0/18 \
  192.168.8.0/24 192.168.227.0/24 10.0.0.0/8 ; do
    ip route replace "$net" dev "$TUN" src "$IP"
done

echo "==> [4] DNS *.ua.pt pelos resolvers da UA via tunel"
resolvectl dns    "$TUN" 192.168.227.20 192.168.227.21
resolvectl domain "$TUN" '~ua.pt'
resolvectl flush-caches 2>/dev/null || true

echo "==> [5] verificacao"
B=$(timeout 8 getent hosts banana.ua.pt | awk '{print $1; exit}')
echo "    banana.ua.pt -> ${B:-NAO_RESOLVIDO}"
[ -n "${B:-}" ] && { ip route replace "$B/32" dev "$TUN" src "$IP"; timeout 5 bash -c "echo >/dev/tcp/$B/22" 2>/dev/null && echo "    banana:22 ABERTA" || echo "    banana:22 falhou"; }
ping -c2 -W2 8.8.8.8 >/dev/null 2>&1 && echo "    internet geral: OK" || echo "    internet geral: SEM resposta"
echo "==> pronto."
