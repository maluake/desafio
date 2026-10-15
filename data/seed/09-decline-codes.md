---
title: Transaction declined on the card machine - common decline/error codes
url: https://www.getnet.net/
lang: en
references:
  - https://www.idinheiro.com.br/negocios/codigos-erro-cartao/
  - https://www.idinheiro.com.br/negocios/erro-05-nao-autorizado/
  - https://blog.pagbank.com.br/o-que-e-codigo-51
---
# Transaction declined (transação negada / não autorizada)

Most declines come from the card issuer (the shopper's bank), not from the machine. The response code on the screen/receipt indicates the cause. Codes are standard in the Brazilian market (ABECS) and are similar across acquirers.

## Common codes

- 05 - Not authorised (não autorizada): generic decline by the issuing bank. Ask the shopper to contact their bank or use another card.
- 51 - Insufficient funds / limit exceeded (saldo ou limite insuficiente). Also used for fraud suspicion. Ask the shopper to use another card or contact the issuer.
- 54 - Expired card (cartão vencido).
- 55 - Wrong PIN (senha inválida). After several attempts the card may be blocked by the issuer.
- 57 - Transaction not allowed for this card (e.g. function credit/debit not enabled). Try the other function or another card.
- 14 - Invalid card number / brand not accepted. Try another card.
- 03 / 11 - Installments not allowed: the card or the merchant plan does not support that number of installments. Reduce the number of installments.
- 26 - Duplicate transaction: the same sale was already registered. Check the sales report before charging again.
- 08 / AA - Timeout (tempo esgotado): communication timed out. Check the connection and try again after a few minutes; do not repeat the sale until you confirm it was not approved.
- 59 - Suspected fraud: the issuer blocked the transaction. The shopper must contact the bank.

## What the merchant should do

1. Read the code on the screen or receipt.
2. If it is an issuer code (05, 51, 54, 55, 57, 59), guide the shopper to contact their bank or use another payment method (Pix, another card).
3. If many different cards are being declined, the problem may be the machine or its connection: restart it, test connectivity and, if it persists, contact Getnet support.
4. Never repeat a sale that timed out before checking the sales report, to avoid double charging.
