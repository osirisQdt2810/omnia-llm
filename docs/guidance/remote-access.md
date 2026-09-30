# Using it from other computers

omnia-llm only listens on the machine it runs on. To use it from Omnia on another computer,
publish it over HTTPS with a tunnel service. This guide uses ngrok, whose free plan includes one
fixed address. The tunnel is opened from your server outward, so nothing on the server has to
accept connections from the internet.

## Publish it with ngrok

1. Sign up at [ngrok.com](https://ngrok.com). From its dashboard, copy your authtoken and your
   free static domain, which is listed under Domains and looks like `<name>.ngrok-free.dev`.
2. Download ngrok for Linux and put the program in this folder as `bin/ngrok`. Then register your
   authtoken: `bin/ngrok config add-authtoken <your-authtoken>`
3. `cp deploy/ngrok.env.example deploy/ngrok.env`, and write your domain in it.
4. Start it with the machine, like the server itself
   ([NVIDIA guide](nvidia.md#run)): `ln -s "$PWD/deploy/systemd/omnia-llm-ngrok.service" ~/.config/systemd/user/`,
   then `systemctl --user daemon-reload` and `systemctl --user enable --now omnia-llm-ngrok`.

Omnia's Base URL is then `https://<your-domain>/v1`.

Always publish on your fixed domain, as the service does. Without one, ngrok makes up an address,
and that address can contain your server's IP.

## Who can use it

Every request needs a token. Give each person or device a token of its own, so that you can cut
one off without affecting the others:

| To | Run |
|---|---|
| Issue a token (it is shown once) | `.venv/bin/omnia-llm token issue <name>` |
| See who has one | `.venv/bin/omnia-llm token list` |
| Cut one off | `.venv/bin/omnia-llm token revoke <name>` |

`scripts/generate_auth_token.py` does the same, as `<name>`, `--list` and `--revoke <name>`.

A revoked token stops working on its next request, with no restart. After ten wrong tokens within
ten minutes, the address that sent them is locked out for fifteen minutes.
