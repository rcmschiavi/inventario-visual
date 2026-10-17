"""Ponto de entrada: ``python -m inventario_visual`` ou ``inventario-visual``."""

from __future__ import annotations

import argparse
import logging
import socket
import sys

import uvicorn

from .app import create_app
from .config import Settings


def lan_ip() -> str:
    """Descobre o IP da máquina na rede local (sem enviar pacotes)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("10.255.255.255", 1))
            return str(s.getsockname()[0])
        except OSError:
            return "127.0.0.1"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Servidor inventario-visual")
    parser.add_argument("--host", help="endereço de escuta (padrão: 0.0.0.0)")
    parser.add_argument("--port", type=int, help="porta HTTPS (padrão: 8443)")
    parser.add_argument(
        "--http",
        action="store_true",
        help="roda sem TLS (só funciona em localhost; o celular exige HTTPS para a câmera)",
    )
    args = parser.parse_args(argv)

    settings = Settings()
    if args.host:
        settings.host = args.host
    if args.port:
        settings.port = args.port

    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    ssl_kwargs: dict[str, str] = {}
    scheme = "http"
    if not args.http:
        if not settings.cert_file.is_file() or not settings.key_file.is_file():
            print(
                f"Certificado não encontrado ({settings.cert_file}, {settings.key_file}).\n"
                "Gere com:  python scripts/gerar_certificado.py",
                file=sys.stderr,
            )
            sys.exit(1)
        ssl_kwargs = {"ssl_certfile": str(settings.cert_file), "ssl_keyfile": str(settings.key_file)}
        scheme = "https"

    print(f"\n  Abra no celular:  {scheme}://{lan_ip()}:{settings.port}/\n")
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        **ssl_kwargs,  # type: ignore[arg-type]
    )


if __name__ == "__main__":
    main()
