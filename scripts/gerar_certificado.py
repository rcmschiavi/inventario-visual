#!/usr/bin/env python3
"""Gera o certificado HTTPS para a rede local.

Prioridade:

1. **mkcert** (se estiver no PATH): cria uma CA local e um certificado para
   ``localhost``, os IPs da máquina e o hostname.
2. **Fallback** com a biblioteca ``cryptography``: cria a mesma estrutura
   (CA própria + certificado do servidor assinado por ela).

Saída em ``certs/``:

* ``cert.pem`` / ``key.pem`` — usados pelo servidor (uvicorn).
* ``rootCA.pem`` / ``rootCA.crt`` — CA a instalar no celular (o servidor também
  a disponibiliza em ``https://<ip>:8443/ca.crt``).

Rode de novo sempre que o IP do PC mudar. Uso::

    python scripts/gerar_certificado.py                 # detecta IPs automaticamente
    python scripts/gerar_certificado.py --ip 192.168.0.50
    python scripts/gerar_certificado.py --sem-mkcert    # força o fallback
"""

from __future__ import annotations

import argparse
import datetime as dt
import ipaddress
import shutil
import socket
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
PASTA_CERTS = RAIZ / "certs"


def ips_locais() -> list[str]:
    """Lista IPv4 da máquina (rota padrão + interfaces conhecidas pelo hostname)."""
    ips: set[str] = {"127.0.0.1"}
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("10.255.255.255", 1))
            ips.add(s.getsockname()[0])
        except OSError:
            pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(str(info[4][0]))
    except socket.gaierror:
        pass
    # Ignora endereços link-local (169.254.x.x), comuns em adaptadores virtuais.
    return sorted(ip for ip in ips if not ip.startswith("169.254."))


def nomes_alvo(ips_extras: list[str]) -> list[str]:
    nomes = ["localhost", "::1", socket.gethostname(), *ips_locais(), *ips_extras]
    vistos: list[str] = []
    for n in nomes:
        if n and n not in vistos:
            vistos.append(n)
    return vistos


# --------------------------------------------------------------------------- #
# mkcert
# --------------------------------------------------------------------------- #


def gerar_com_mkcert(mkcert: str, nomes: list[str]) -> None:
    # -install coloca a CA no repositório de confiança do PC (útil para testar
    # no navegador do próprio PC). Pode pedir confirmação no Windows.
    subprocess.run([mkcert, "-install"], check=False)
    subprocess.run(
        [
            mkcert,
            "-cert-file",
            str(PASTA_CERTS / "cert.pem"),
            "-key-file",
            str(PASTA_CERTS / "key.pem"),
            *nomes,
        ],
        check=True,
    )
    caroot = subprocess.run(
        [mkcert, "-CAROOT"], check=True, capture_output=True, text=True
    ).stdout.strip()
    shutil.copyfile(Path(caroot) / "rootCA.pem", PASTA_CERTS / "rootCA.pem")


# --------------------------------------------------------------------------- #
# Fallback com cryptography
# --------------------------------------------------------------------------- #


def gerar_com_cryptography(nomes: list[str]) -> None:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    agora = dt.datetime.now(dt.UTC)
    ca_key_path = PASTA_CERTS / "rootCA-key.pem"
    ca_path = PASTA_CERTS / "rootCA.pem"

    # Reaproveita a CA existente para não precisar reinstalar no celular
    # quando só o IP mudar.
    if ca_key_path.is_file() and ca_path.is_file():
        ca_key = serialization.load_pem_private_key(ca_key_path.read_bytes(), password=None)
        ca_cert = x509.load_pem_x509_certificate(ca_path.read_bytes())
        print("Reaproveitando CA existente em certs/rootCA.pem")
    else:
        ca_key = ec.generate_private_key(ec.SECP256R1())
        ca_nome = x509.Name(
            [
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "inventario-visual"),
                x509.NameAttribute(NameOID.COMMON_NAME, f"inventario-visual CA ({socket.gethostname()})"),
            ]
        )
        ca_cert = (
            x509.CertificateBuilder()
            .subject_name(ca_nome)
            .issuer_name(ca_nome)
            .public_key(ca_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(agora - dt.timedelta(days=1))
            .not_valid_after(agora + dt.timedelta(days=3650))
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True, key_cert_sign=True, crl_sign=True,
                    content_commitment=False, key_encipherment=False, data_encipherment=False,
                    key_agreement=False, encipher_only=False, decipher_only=False,
                ),
                critical=True,
            )
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False)
            .sign(ca_key, hashes.SHA256())  # type: ignore[arg-type]
        )
        ca_key_path.write_bytes(
            ca_key.private_bytes(  # type: ignore[union-attr]
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
        ca_path.write_bytes(ca_cert.public_bytes(serialization.Encoding.PEM))

    # Certificado do servidor. iOS exige validade <= 825 dias, SAN e EKU serverAuth.
    sans: list[x509.GeneralName] = []
    for n in nomes:
        try:
            sans.append(x509.IPAddress(ipaddress.ip_address(n)))
        except ValueError:
            sans.append(x509.DNSName(n))
    key = ec.generate_private_key(ec.SECP256R1())
    cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "inventario-visual")]))
        .issuer_name(ca_cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(agora - dt.timedelta(days=1))
        .not_valid_after(agora + dt.timedelta(days=397))
        .add_extension(x509.SubjectAlternativeName(sans), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_cert.public_key()),  # type: ignore[arg-type]
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())  # type: ignore[arg-type]
    )
    (PASTA_CERTS / "key.pem").write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    (PASTA_CERTS / "cert.pem").write_bytes(cert.public_bytes(serialization.Encoding.PEM))


def exportar_ca_der() -> None:
    """Gera ``rootCA.crt`` em DER, formato aceito tanto por Android quanto por iOS."""
    from cryptography import x509
    from cryptography.hazmat.primitives import serialization

    ca = x509.load_pem_x509_certificate((PASTA_CERTS / "rootCA.pem").read_bytes())
    (PASTA_CERTS / "rootCA.crt").write_bytes(ca.public_bytes(serialization.Encoding.DER))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ip", action="append", default=[], help="IP/nome extra (pode repetir)")
    parser.add_argument("--sem-mkcert", action="store_true", help="não usa mkcert mesmo se instalado")
    args = parser.parse_args()

    PASTA_CERTS.mkdir(exist_ok=True)
    nomes = nomes_alvo(args.ip)
    print("Nomes/IPs no certificado:", ", ".join(nomes))

    mkcert = None if args.sem_mkcert else shutil.which("mkcert")
    if mkcert:
        print(f"Usando mkcert ({mkcert})")
        try:
            gerar_com_mkcert(mkcert, nomes)
        except (subprocess.CalledProcessError, OSError) as exc:
            print(f"mkcert falhou ({exc}); usando fallback com cryptography.", file=sys.stderr)
            gerar_com_cryptography(nomes)
    else:
        print("mkcert não encontrado; gerando CA própria com a biblioteca cryptography.")
        gerar_com_cryptography(nomes)

    exportar_ca_der()
    print(
        "\nPronto! Arquivos em certs/:\n"
        "  cert.pem, key.pem   -> usados pelo servidor\n"
        "  rootCA.crt          -> instale no celular (veja o README)\n"
        "  (rootCA-key.pem, se existir, é SECRETO: não compartilhe)\n"
    )


if __name__ == "__main__":
    main()
