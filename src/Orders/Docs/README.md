# Fulfillment Transaction Service (OrderSaga)

Back to platform overview: [Root README](../../../README.md)

## Business Requirement

Northstar checkout must coordinate three external systems safely:

1. inventory reservation
2. payment capture
3. shipment creation

If any stage fails, operations must receive enough state to perform controlled rollback actions.

## Implemented Business Rules

Source: `src/OrderSaga/Pipelines/OrderSagaPipeline.cs`, `src/OrderSaga/Steps/`

1. Reserve inventory before charging payment.
2. Charge payment before creating shipment.
3. Stop execution at first failure.
4. Preserve identifiers (`ReservationId`, `ChargeAmount`, `ShipmentId`) needed for cleanup.

## Features Demonstrated

**EvalApp pattern:** Sequential async steps with rollback identifiers in data contract

**SOLID principle:** DIP (steps depend on service interfaces, not implementations)

## Implementation


| Concern | Path |
|---|---|
| Pipeline orchestration | `src/OrderSaga/Pipelines/OrderSagaPipeline.cs` |
| Transaction steps | `src/OrderSaga/Steps/` |
| Service interfaces | `src/OrderSaga/Services/` |
| Executable specs | `Tests/Features/OrderSaga/` |


Verify: `dotnet test --filter "OrderSaga"`
